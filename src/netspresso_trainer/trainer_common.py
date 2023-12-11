from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataloaders import build_dataloader, build_dataset
from .models import SUPPORTING_TASK_LIST, build_model, is_single_task_model
from .pipelines import build_pipeline
from .utils.environment import set_device
from .utils.logger import add_file_handler, get_logging_dir, set_logger


def train_common(conf: DictConfig, log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'):

    assert bool(conf.model.fx_model_checkpoint) != bool(conf.model.checkpoint)
    is_graphmodule_training = bool(conf.model.fx_model_checkpoint)

    distributed, world_size, rank, devices = set_device(conf.training.seed)
    logger = set_logger(level=log_level, distributed=distributed)

    conf.distributed = distributed
    conf.world_size = world_size
    conf.rank = rank

    task = str(conf.model.task).lower()
    assert task in SUPPORTING_TASK_LIST

    # TODO: Get model name from checkpoint
    single_task_model = is_single_task_model(conf.model)
    conf.model.single_task_model = single_task_model

    model_name = str(conf.model.name).lower()

    if is_graphmodule_training:
        model_name += "_graphmodule"

    logging_dir: Path = get_logging_dir(task, model_name, output_root_dir="./outputs", distributed=distributed)
    add_file_handler(logging_dir / "result.log", distributed=conf.distributed)

    if not distributed or dist.get_rank() == 0:
        logger.info(f"Task: {task} | Model: {model_name} | Training with torch.fx model? {is_graphmodule_training}")
        logger.info(f"Result will be saved at {logging_dir}")

    if conf.distributed and conf.rank != 0:
        torch.distributed.barrier()  # wait for rank 0 to download dataset

    train_dataset, valid_dataset, test_dataset = build_dataset(conf.data, conf.augmentation, task, model_name, distributed=distributed)

    if conf.distributed and conf.rank == 0:
        torch.distributed.barrier()

    train_dataloader, eval_dataloader = \
        build_dataloader(conf, task, model_name, train_dataset=train_dataset, eval_dataset=valid_dataset)

    if is_graphmodule_training:
        assert conf.model.fx_model_checkpoint is not None
        assert Path(conf.model.fx_model_checkpoint).exists()
        model = torch.load(conf.model.fx_model_checkpoint)
    else:
        model = build_model(conf.model, task, train_dataset.num_classes, conf.model.checkpoint, conf.augmentation.img_size)

    model = model.to(device=devices)
    if conf.distributed:
        model = DDP(model, device_ids=[devices], find_unused_parameters=True)  # TODO: find_unused_parameters should be false (for now, PIDNet has problem)

    trainer = build_pipeline(conf, task, model_name, model,
                             devices, train_dataloader, eval_dataloader,
                             class_map=train_dataset.class_map,
                             logging_dir=logging_dir,
                             is_graphmodule_training=is_graphmodule_training)

    trainer.set_train()
    try:
        trainer.train()

        if test_dataset:
            trainer.inference(test_dataset)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
