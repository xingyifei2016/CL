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

LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)
xent = torch.nn.CrossEntropyLoss()
def loss_fn(x, y):
    return xent(x, y), (torch.argmax(x, dim=-1) == y).type(torch.FloatTensor).sum()

    
def eval_model(val_loader, net, current_iter, tracker, writer, logger):
    net.eval()
    logger.info("Validating....")
    val_losses = []
    val_acc = []

    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(val_loader), dynamic_ncols=True):
            
            x, y = batch
            loss, acc = loss_fn(net(x), y)
            print(net(x).shape)
            print(torch.argmax(net(x), dim=-1))
            print(y)
            val_losses.append(loss.item())
            val_acc.append(acc.item())
            print("ACC: "+str(acc.item()))
    

    mean_val_acc = np.sum(val_acc)/len(val_loader.dataset)
    mean_val_loss = np.sum(val_losses)/len(val_loader.dataset)
    writer.add_scalar('Validation Loss', mean_val_loss, current_iter)
    writer.add_scalar('Validation Acc', mean_val_acc, current_iter)
    logger.info("Finished Validation! Mean Loss: {}, Acc: {}".format(
        mean_val_loss, mean_val_acc))
    return tracker.update(mean_val_acc)



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
    num_tasks = args.num_tasks
    
    dset_train = os.path.join(dset_path, 'train')
    dset_test = os.path.join(dset_path, 'new_test')

    
    # Tracking best validation model
    tracker = logger.Model_Tracker(ckpt_dir, LOG)
    save_this_iter = False

    task_info = dataset.create_tasks(num_tasks)
    
    for i in range(len(task_info)):
        train_dataset = dataset.tinyImageNet(root=dset_train, subset=task_info[i])
        dset_train_val = dataset.train_val_dataset(train_dataset)
        
        train_loader = torch.utils.data.DataLoader(dset_train_val['train'], batch_size=args.bs, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dset_train_val['val'], batch_size=args.bs, shuffle=False)
        
        test_dataset = dataset.tinyImageNet(root=dset_test, subset=task_info[i])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
        
        for current_iter in range(epoch):
            eval_model(val_loader, net, current_iter, tracker, writer, LOG)
            eval_model(test_loader, net, current_iter, tracker, writer, LOG)
        
        
        

    LOG.info("Done. Exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="global random seed.")
    
    parser.add_argument("--checkpoint_dir", default='./ckpt/',
                        help="Output directory where checkpoints are saved")
    
    parser.add_argument("--dset_path", default='/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200/',
                        help="Directory where data is stored")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer")
    
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to incrementally perform")
    
    parser.add_argument("--bs", type=int, default=256, help="Batch size")

    parser.add_argument("--wd", type=float, default=0, help="Weight Decay")

    parser.add_argument("--epoch_per_task", type=int, default=5,
                        help="Epochs per task")

    parser.add_argument("--deterministic", action='store_true', default=False,
                        help="Use torch's deterministic algorithms. Breaks some models")

    parser.add_argument("--note", type=str, default=None,
                        help="any additional notes or remarks")

    parser.set_defaults()

    args = parser.parse_args()

    main(args)