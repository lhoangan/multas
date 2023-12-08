#!/bin/bash -l

# `-l`: make bash act as if it had been invoked as a login shell:
# The argument is indispensable for activating commands `.bashrc`
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# Here placed the commands SBATCH defining the parameters for node reservation
# On peut éventuellement placer ici les commentaires SBATCH permettant de
# définir les paramètres par défaut de lancemen:

# asdf#SBATCH -p shortrun

#SBATCH --time 2-00:00:00
#SBATCH --exclude=sn[2-7]
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=FAIL,END
#SBATCH --mem-per-cpu=4G

# Spécification de la version CUDA
# setcuda 11.6

# Activation de l'environnement anaconda python37

conda activate bmvc22
# Exécution du script habituellement utilisé, on utilise la variable
# CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservé
export PYTHONPATH=/share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/pycocotools/:$PYTHONPATH

export WANDB_MODE=online
export WANDB_PROJECT=tw_eccv24

params="${@:1}"

# Perform basic parsing to (1) create unique saving weight_paths (2) input to student
# DEFAULT parameters
task_t="det"
match_t="iou"
conf_loss_t="fc"
# [ref] https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--match)
        match_t="$2"              ; shift ; shift ;; # past argument ; past value ; end clause
    -cl|--conf_loss)
        conf_loss_t="$2"          ; shift ; shift ;;
    -bb|--backbone)
        backbone_t="$2"         ; shift ; shift ;;
    -n|--neck)
        neck_t="$2"             ; shift ; shift ;;
    -d|--dataset)
        dataset_t="$2"          ; shift ; shift ;;
    -is|--imgset)
        imgset_t="$2"           ; shift ; shift ;;
    -t|--task)
        task_t="$2"             ; shift ; shift ;;
    --task_s)
        task_s="--task $2"      ; shift ; shift ;;
    --g|--group)
        group="--group $2"      ; shift ; shift ;;
    *)
      shift                     # past argument
  esac
done

# set teacher backbone from student backbone
case ${backbone_t} in
    'resnet50')         backbone_s='resnet18'         ;;
    'vgg16')            backbone_s='vgg11'            ;;
    'shufflenet-1.0')   backbone_s='shufflenet-0.5'   ;;
    'repvgg-A2')        backbone_s='repvgg-A0'        ;;
    'regnet800')        backbone_s='regnet400'        ;;
  # 'repvgg-A1')        backbone_t='repvgg-A2' ;;
esac
neck_s="fpn"
dataset_s="VOC"
imgset_s="Main"
batch_size=10

case ${dataset_t} in
    '*VOC*')
        base_anchor_size=24
        size=320
    ;;
    '*MXE*')
        base_anchor_size=24
        size=320
    ;;
    # 'VEDAI?') # VEDAI, VEDAI2
    #     base_anchor_size=16 #     size=512 #     batch_size=5 #     dataset_s="VEDAI" # ;;
    # 'xview')
    #     base_anchor_size=8 #     size=512 # ;;
    # 'VEDAI1024')
    #     base_anchor_size=32 #     size=512 #     batch_size=5 # ;;
esac

# Not real WANDB environmental variable
# storing teacher trained weight to be used when training student
date=$(date +'%m%d').$(date +"%H%M")
mkdir weight_paths
export WANDB_EXP=weight_paths/${dataset_t}_${imgset_t}-${task_t}-${backbone_t}_${neck_t}-${match_t}_${conf_loss_t}-${date}
echo "Trained weights are to stored here: "$WANDB_EXP

# ------------------------
# TODO: to be removed
export WANDB_MODE=online
# ------------------------

export WANDB_PROJECT=ICCVw23-Revived
set -x
python train.py --seed 0 --eval_epoch 1 --lr 0.005 --batch_size 7   \
                --match mg --conf_loss gfc  ${group}                \
                ${params} && {

    echo "Finish training teacher. Trained weights are at: "
    cat $WANDB_EXP

    echo -e "\nScheduling student for training: " # -e for displying \n
    set -x

    kdc_s=("soft")
    case ${imgset_t} in
        'Half')
            imgsets=("Main" "Half2")
            kdc_s=("soft" "hard")
        ;;
        'Quarter')
            imgsets=("Half2" "3Quarter")
        ;;
        'Eighth')
            imgsets=("Half2" "3Quarter" "7Eighth")
        ;;
    esac

    for imgset_s in ${imgsets[@]}; do
        for kdc in ${kdc_s[@]} ; do
            for kdf in "" "+mse" "+pdf" "+defeat"; do
                kd=${kdc}${kdf}

                sbatch runs/distil.sh  --seed 0 --eval_epoch 1 --batch_size 10 \
                    --conf_loss gfc    --lr 0.01                               \
                    --double_aug --nepoch 70  --job_type student     ${group}  \
                    --kd                ${kd}                        ${task_s} \
                    --dataset           ${dataset_s}  --imgset       ${imgset_s}\
                    --backbone          ${backbone_s} --neck         ${neck_s} \
                    --teacher_backbone  ${backbone_t} --teacher_neck ${neck_t} \
                    --t${task_t:0:3}_weights $(cat $WANDB_EXP)

                sleep 5m
            done

            sleep 10m
        done

        sleep 15m
    done

} || {

    echo "Training teacher FAILED!"
    rm $WANDB_EXP

}
