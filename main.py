import dataset
import logger

import argparse
import config
import os
import sys
import logging
import coloredlogs
from pdb import set_trace as st

import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from models import EWC

LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)
xent = torch.nn.CrossEntropyLoss()

def main(args):
    # cfg = config.parse_config(args.config)

    ckpt_dir = args.checkpoint_dir
    logger.setup_ckpt(ckpt_dir)
    logger.setup_logger(LOG, ckpt_dir)
    # LOG.info(f"\ncfg: {str(cfg)}")
    LOG.info(f"args: {str(args)}")
    LOG.info(f"system command: {''.join(sys.argv)}")
    epoch = args.epoch_per_task

    # Tensorboard
    writer = SummaryWriter(ckpt_dir+'/runs')

    ##### Random seed #####
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    #######################    
    
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # First initialize placeholder model

    # net.cuda()
    net.zero_grad()
    net.train()

    LOG.info(f"# Params: {logger.count_params(net)}")

    optimizer = torch.optim.AdamW(
        params=net.parameters(), lr=args.lr, weight_decay=args.wd)
    
    ##### Process dataset #####
    dset_path = args.dset_path
    if not os.path.exists(dset_path):
        # If path does not exist, then create a dataset on this path
        dataset.create_dir(dset_path)
        dataset.download_tinyImagenet(dset_path)
        dataset.reorganize_files()
    num_tasks = args.num_tasks
    
    # Tracking best validation model
    tracker = logger.Model_Tracker(ckpt_dir, LOG)
    task_info = dataset.create_tasks(num_tasks)
    
    # Per task dataset construction, training, validation, and testing.
    for task_number in range(len(task_info)):
        dataset_dict = prepare_loaders(dset_path, task_info, task_number, args)
        EWC.per_task_updates(net, dataset_dict, writer, LOG, task_info, args, task_number)

    LOG.info("Done. Exiting")

    
def loss_fn(x, y):
    return xent(x, y), (torch.argmax(x, dim=-1) == y).type(torch.FloatTensor).sum()

def prepare_loaders(dset_path, task_info, task_number, args):
    dset_train = os.path.join(dset_path, 'train')
    dset_test = os.path.join(dset_path, 'new_test')
    train_dataset = dataset.tinyImageNet(root=dset_train, subset=task_info[task_number])
    test_dataset = dataset.tinyImageNet(root=dset_test, subset=task_info[task_number])
    dset_train_val = dataset.train_val_dataset(train_dataset)
    
    res_dict = {}
    
    res_dict['train'] = torch.utils.data.DataLoader(dset_train_val['train'], batch_size=args.bs, shuffle=True)
    res_dict['val'] = torch.utils.data.DataLoader(dset_train_val['val'], batch_size=args.bs, shuffle=True)
    res_dict['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=True)
    
    # This step is used for certain methods which rely on previous trained data
    if task_number != 0:
        prev_train_dataset = dataset.tinyImageNet(root=dset_train, subset=task_info[task_number-1])
        prev_dset_train_val = dataset.train_val_dataset(prev_train_dataset)
        res_dict['prev_train_dataset'] = torch.utils.data.DataLoader(prev_dset_train_val['train'], batch_size=args.bs, shuffle=True)
        
    return res_dict
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="global random seed.")
    
    parser.add_argument("--checkpoint_dir", default='./ckpt/',
                        help="Output directory where checkpoints are saved")
    
    parser.add_argument("--dset_path", default='/home/xingyifei/CL/dataset/tiny-imagenet-200',
                        help="Directory where data is stored")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    
    parser.add_argument("--num_epochs", type=float, default=1,
                        help="Learning rate for the optimizer")
    
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to incrementally perform")
    
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    
    parser.add_argument("--mt", type=float, default=0.9, help="Momentum")
    
    parser.add_argument("--wd", type=float, default=0.001, help="Weight Decay")

    parser.add_argument("--epoch_per_task", type=int, default=5,
                        help="Epochs per task")

    parser.add_argument("--deterministic", action='store_true', default=False,
                        help="Use torch's deterministic algorithms. Breaks some models")

    parser.add_argument("--note", type=str, default=None,
                        help="any additional notes or remarks")

    parser.set_defaults()

    args = parser.parse_args()

    main(args)