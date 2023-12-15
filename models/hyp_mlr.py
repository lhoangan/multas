# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import geoopt


EPSILON = 1e-5


def expmap0_feature_map(ball: geoopt.PoincareBall, z: torch.Tensor) -> torch.Tensor:
    """
    Embeds the image feature map in the Poincaré ball.
    Parameters
    ==========
        ball: geoopt.PoincareBall
        z: torch.Tensor - size (b,d,w,h) Euclidean features
    Returns
    =======
        z: torch.Tensor - size (b,d,w,h) Poincaré ball features
    """
    z = z.permute(0,2,3,1)
    z = ball.expmap0(z)
    z = z.permute(0,3,1,2)
    return z


class PoincareProjector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ball = geoopt.PoincareBall()
    def forward(self, x):
        return expmap0_feature_map(self.ball, x)


def hyp_dist(p: torch.Tensor, w: torch.Tensor, z: torch.Tensor, c: float) -> torch.Tensor:
    """
    Pairwise Hyperbolic distance between the Gyroplan H(p, w) and z.
    Parameters
    ==========
        p: torch.Tensor - size (p,d) points in the Poincare ball
        w: torch.Tensor - size (p,d) normal vectors
        z: torch.Tensor - size (b,d,w,h) pixel vectors already in the Poincare ball
        c: float - curvature
    Returns
    =======:
        D: torch.Tensor - size (b,p,w,h) distances
    """
    c = torch.tensor(c, device=p.device)
    c_sqrt = c.sqrt()

    p_hat = -p

    p_dot_z = torch.einsum("pd,bdwh->bpwh", p_hat, z) # (b,p,w,h)
    z_norm = z.pow(2).sum(1, keepdim=True) # (b,1,w,h)
    p_norm = p.pow(2).sum(-1).view(1,-1,1,1) # (1,p,1,1)
    w_norm = w.pow(2).sum(-1).view(1,-1,1,1) # (1,p,1,1)

    # conformal factor at point p with curvature c
    # lambda_p_c = 2. / (1 - c * p_norm) # (1,p,1,1)
    lambda_p_c = 2. # value used in the paper

    denom = (1 + 2 * c * p_dot_z + c ** 2 * z_norm * p_norm).clamp(min=EPSILON)
    alpha = (1 + 2 * c * p_dot_z + c * z_norm) / denom
    beta = (1 - c * p_norm) / denom

    p_dot_w = torch.einsum("pd,pd->p", p_hat, w).view(1,-1,1,1)
    z_dot_w = torch.einsum("bdwh,pd->bpwh", z, w)

    p_m_y_w = alpha * p_dot_w + beta * z_dot_w
    p_m_y_w_norm = alpha ** 2 * p_norm + 2 * alpha * beta * p_dot_z + beta ** 2 * z_norm

    logits = (
        (lambda_p_c * w_norm / c_sqrt) *
        torch.asinh((2 * c_sqrt * p_m_y_w / ((1 - c * p_m_y_w_norm) * w_norm).clamp(min=EPSILON)).clamp(max=85.))
    )

    return logits


class HyperbolicLayer(nn.Module):
    "Hyperbolic MLR, Ganea et al. 2018"
    def __init__(self, n_classes: int, dims: int, c: float = 1.):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)

        dirs = self.ball.expmap0(torch.randn(n_classes, dims))
        self.register_parameter("p", geoopt.ManifoldParameter(dirs, manifold=self.ball))

        # Normals live in the euclidean space.
        normals = torch.nn.functional.normalize(dirs, dim=-1)
        self.register_parameter("normals", nn.Parameter(normals))
        self.c = c

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ==========
            z: Tensor - size (b,d,h,w)
        Returns
        =======
            logits: Tensor - size (b,n_classes,h,w)
        """
        z = expmap0_feature_map(self.ball, z)
        logits = -hyp_dist(self.p, self.normals, z, c=self.c)
        return logits


def pairwise_busemann(
    p: torch.Tensor,
    a: torch.Tensor,
    z: torch.Tensor,
    c: float = 1.,
    lambda_: float = 0.,
) -> torch.Tensor:
    """
    Parameters
    ==========
        p: torch.Tensor - size (p,d) protos
        a: torch.Tensor - size (p,) bias
        z: torch.Tensor - size (b,d,w,h) points
        c: float - curvature
        lambda_: float - regularization
    Returns
    =======
        B: torch.Tensor - size (b,p,w,h) Busemann distance
    """
    sc = torch.tensor(c, device=p.device).sqrt()

    # Not very memory efficient...
    D = (p[None,:,:,None,None] - sc * z[:,None,:,:,:]).pow(2).sum(2) # (b,p,w,h)
    nz = z.pow(2).sum(1, keepdim=True) # (b,1,w,h)

    B = torch.log(
        (D / (1 - c * nz).clamp(min=EPSILON)).clamp(min=EPSILON)
    ) / sc 

    reg = lambda_ * torch.log((1 - c * nz).clamp(min=EPSILON))

    return (-B + a.view(1,-1,1,1)) + reg


class HorosphericalLayer(nn.Module):
    "Busemann prototypes with trainable prototype positions"
    def __init__(self, protos, lambda_: float = 0., c: float = 1.):
        super().__init__()
        assert isinstance(protos, torch.Tensor)
        self.ball = geoopt.PoincareBall(c=c)
        self.register_parameter(
            "protos",
            geoopt.ManifoldParameter(protos, manifold=geoopt.Sphere()),
        )
        bias = torch.zeros(protos.size(0))
        self.register_parameter("bias", torch.nn.Parameter(bias))
        assert c == 1.
        self.lambda_ = lambda_
        self.phi = self.lambda_ * self.protos.size(-1)
        self.c = c

    def forward(self, x):
        """
        Parameters
        ==========
            x: torch.Tensor - size (b,d,w,h)
        Returns
        =======
            logits: torch.Tensor - size (b,p,w,h)
        """
        x = expmap0_feature_map(self.ball, x)
        return pairwise_busemann(
            self.protos,
            self.bias,
            x,
            c=self.c,
            lambda_=self.phi,
        )


class BusemannLoss(nn.Module):
    def __init__(self, protos, lambda_: float = 0., c: float = 1.):
        super().__init__()
        assert isinstance(protos, torch.Tensor)
        self.ball = geoopt.PoincareBall(c=c)
        self.register_buffer("protos", protos)
        self.lambda_ = lambda_
        self.register_buffer("c", torch.tensor(c))
        self.register_buffer("sc", torch.tensor(c).sqrt())
        self.register_buffer("k", torch.tensor(-c))

    def forward(self, x, targets):
        # x (b,d,w,h)
        # y (b, w,h)
        d = x.size(1)
        assert d == self.protos.size(-1)

        c = self.c
        sc = self.sc

        # remove samples from these classes
        mask = (targets != 255) & (targets != -1)

        x = x.permute(0,2,3,1) # (b,h,w,d)
        x = x[mask]
        x = x.reshape(-1,d) # (b*h*w,d)

        # Embed in hyperbolic space
        x = self.ball.expmap0(x)

        targets = targets[mask]

        yp = self.protos[targets].reshape(-1,d) # (b*h*w,d)
        nx = x.pow(2).sum(-1)

        assert torch.all(nx <= 1.)

        denom = (1. - c * nx).clamp(min=EPSILON)
        reg = self.lambda_ * torch.log(denom)

        return (
            torch.log(
                ((yp - sc * x).pow(2).sum(-1) /
                denom).clamp(min=EPSILON)
            ) / sc - reg
        ).mean()


class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()
