import argparse
import os
import logging

import torch
import numpy as np
import torch.distributed as dist

from ECPE.utils import parse_args, str2bool
from ECPE.data_reader import DocumentDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
_logger = logging.getLogger('main')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_distributed', type=str2bool, default=False,
                        help='Whether to run distributed training.')
    parser.add_argument('--save_path', type=str, default='output',
                        help='The path where to save models.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='The random seed to control the data generation.')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])

    args = parse_args(parser)
    return args


def main(args):
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1
    if args.is_distributed and int(os.environ['WORLD_SIZE']) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
    else:
        args.is_distributed = False
        _logger.info('Running with a single process on GPU.')

    if args.local_rank == 0:
        args.display()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed + args.rank)

    if args.phase == 'train':
        train_set = DocumentDataset(phase=args.phase)


if __name__ == '__main__':
    args = setup_args()
    main(args)
