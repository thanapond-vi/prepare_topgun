"""Deterministic BGR experiment; use these adapters for training AND evaluation.

Ultralytics normally creates RGB tensors. Swapping channels once makes the model
see the OpenCV BGR channel order used by the reference Kaggle CNN notebook.
Weights alone do not encode this preprocessing requirement.
"""
from copy import copy

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator


def swap_rgb_bgr(batch):
    batch['img'] = batch['img'].flip(1).contiguous()
    return batch


class BGRSegmentationValidator(SegmentationValidator):
    def preprocess(self, batch):
        return swap_rgb_bgr(super().preprocess(batch))


class BGRSegmentationTrainer(SegmentationTrainer):
    def preprocess_batch(self, batch):
        return swap_rgb_bgr(super().preprocess_batch(batch))

    def get_validator(self):
        return BGRSegmentationValidator(
            self.test_loader, save_dir=self.save_dir,
            args=copy(self.args), _callbacks=self.callbacks,
        )
