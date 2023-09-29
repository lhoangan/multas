#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(
        self,
        loader,
    ) -> None:
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(
        self,
    ) -> None:
        try:
            next_ = next(self.loader)
            if len(next_) == 3:
                self.next_input, self.next_target, self.next_seggt = next_
            elif len(next_) == 2:
                self.next_input, self.next_target = next_
                self.next_seggt = None
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_seggt = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = [anno.cuda(non_blocking=True) for anno in self.next_target]
            if self.next_seggt is not None:
                self.next_seggt = self.next_seggt.cuda(non_blocking=True)

    def next(
        self,
    ) -> list:
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        seggt = self.next_seggt
        if input is not None:
            self.record_stream(input)
        if target is not None:
            for t in target:
                t.record_stream(torch.cuda.current_stream())
        if seggt is not None:
            self.record_stream(seggt)
        self.preload()
        if seggt is not None:
            return input, target, seggt
        else:
            return input, target

    @staticmethod
    def _record_stream_for_image(
        input,
    ) -> None:
        input.record_stream(torch.cuda.current_stream())

