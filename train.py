import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from datasets import build_dataset, build_dataloader
from models import build_model
from pipelines import ClassificationPipeline, SegmentationPipeline
from utils.environment import set_device

SUPPORT_TASK = ['classification', 'segmentation']

def parse_args_netspresso():

    parser = argparse.ArgumentParser(description="Parser for NetsPresso configuration")
    
    # -------- User arguments ----------------------------------------
    
    parser.add_argument(
        '--config', type=str, default='',
        dest='config', 
        help="Config path")
    
    parser.add_argument(
        '-o', '--output_dir', type=str, default='..',
        dest='output_dir', 
        help="Checkpoint and result saving directory")

    args, _ = parser.parse_known_args()    
    
    return args

def train():
    args_parsed = parse_args_netspresso()
    args = OmegaConf.load(args_parsed.config)
    distributed, world_size, rank, devices = set_device(args)
    
    args.distributed = distributed
    args.world_size = world_size
    args.rank = rank
    
    task = str(args.train.task).lower()
    assert task in SUPPORT_TASK
    
    if args.distributed and args.rank != 0:
        torch.distributed.barrier() # wait for rank 0 to download dataset
        
    train_dataset, eval_dataset = build_dataset(args)
    
    if args.distributed and args.rank == 0:
        torch.distributed.barrier()
        
    model = build_model(args, train_dataset.num_classes)
    
    train_dataloader, eval_dataloader = \
        build_dataloader(args, model, train_dataset=train_dataset, eval_dataset=eval_dataset)

    model = model.to(device=devices)
    if task == 'classification':
        trainer = ClassificationPipeline(args, model, devices, train_dataloader, eval_dataloader)
    elif task == 'segmentation':
        trainer = SegmentationPipeline(args, model, devices, train_dataloader, eval_dataloader)
    else:
        raise AssertionError(f"No such task! (task: {task})")
    
    trainer.set_train()
    trainer.train()
    
if __name__ == '__main__':
    train()