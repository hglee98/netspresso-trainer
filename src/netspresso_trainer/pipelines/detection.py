import os
import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from .base import BasePipeline
from ..optimizers import build_optimizer
from ..schedulers import build_scheduler
from ..utils.logger import set_logger

logger = logging.getLogger("netspresso_trainer")


class DetectionPipeline(BasePipeline):
    def __init__(self, conf, task, model_name, model, devices, train_dataloader, eval_dataloader, class_map, **kwargs):
        super(DetectionPipeline, self).__init__(conf, task, model_name, model, devices,
                                                train_dataloader, eval_dataloader, class_map, **kwargs)
        self.num_classes = train_dataloader.dataset.num_classes

    def set_train(self):

        assert self.model is not None
        self.optimizer = build_optimizer(self.model,
                                         opt=self.conf.training.opt,
                                         lr=self.conf.training.lr,
                                         wd=self.conf.training.weight_decay,
                                         momentum=self.conf.training.momentum)
        self.scheduler, _ = build_scheduler(self.optimizer, self.conf.training)

    def train_step(self, batch):
        self.model.train()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]

        self.optimizer.zero_grad()
        out = self.model(images, targets=targets)
        self.loss(out, target=targets, mode='train')

        self.loss.backward()
        self.optimizer.step()

        # TODO: metric update
        # out = {k: v.detach() for k, v in out.items()}
        # self.metric(out['pred'], target=targets, mode='train')

        if self.conf.distributed:
            torch.distributed.barrier()

    def valid_step(self, batch):
        self.model.eval()
        images, labels, bboxes = batch['pixel_values'], batch['label'], batch['bbox']
        images = images.to(self.devices)
        targets = [{"boxes": box.to(self.devices), "labels": label.to(self.devices)}
                   for box, label in zip(bboxes, labels)]

        out = self.model(images, targets=targets)
        self.loss(out, target=targets, mode='valid')

        # TODO: metric update
        # self.metric(out['pred'], (labels, bboxes), mode='valid')

        if self.conf.distributed:
            torch.distributed.barrier()

        logs = {
            'images': images.detach().cpu().numpy(),
            'target': [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                       for bbox, label in zip(bboxes, labels)],
            'pred': [(np.concatenate((bbox.detach().cpu().numpy(), confidence.detach().cpu().numpy()[..., np.newaxis]), axis=-1),
                      label.detach().cpu().numpy())
                     for bbox, confidence, label in zip(out['post_boxes'], out['post_scores'], out['post_labels'])],
        }
        return {k: v for k, v in logs.items()}

    def test_step(self, batch):
        self.model.eval()
        images = batch['pixel_values']
        images = images.to(self.devices)

        out = self.model(images.unsqueeze(0))

        results = [(bbox.detach().cpu().numpy(), label.detach().cpu().numpy())
                   for bbox, label in zip(out['post_boxes'], out['post_labels'])],

        return results

    def get_metric_with_all_outputs(self, outputs):
        targets = np.empty((0, 4))
        preds = np.empty((0, 5))  # with confidence score
        targets_indices = np.empty(0)
        preds_indices = np.empty(0)
        for output_batch in outputs:
            for detection, class_idx in output_batch['target']:
                targets = np.vstack([targets, detection])
                targets_indices = np.append(targets_indices, class_idx)

            for detection, class_idx in output_batch['pred']:
                preds = np.vstack([preds, detection])
                preds_indices = np.append(preds_indices, class_idx)

        self.metric((preds, preds_indices), (targets, targets_indices), mode='valid')
        return (preds, preds_indices), (targets, targets_indices)
