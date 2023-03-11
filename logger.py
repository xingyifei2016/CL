import logging

import logging
import coloredlogs
import os
import torch


def count_params(model): return sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)

def setup_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        print("FOLDER EXISTS")
    os.system('rm -r '+ckpt_dir)
    os.makedirs(ckpt_dir+'/runs', exist_ok=True)
    os.makedirs(ckpt_dir+'/code', exist_ok=True)
    # Makes a copy of all local code files to the log directory, just in case there are any problems
    os.system('cp *.py '+ckpt_dir+'/code/')
    # Makes a copy of all config files
    os.system('cp -r config '+ckpt_dir+'/code/')


LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)


def setup_logger(LOG, checkpoint_dir, debug=True):
    '''set up logger'''

    formatter = logging.Formatter('%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    LOG.setLevel(logging.DEBUG)

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "run.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    LOG.addHandler(fh)


class Model_Tracker:
    # Model writer for class incremental models
    
    def __init__(self, path, LOG):
        """
        the files will be saved in the form of parent_dir / task_id / model.ckpt
        each model saved at the particular task_id will be the one with highest validation accuracy 
        
        path is the parent directory which the models should be saved
        LOG is the logger that will produce
        tasks is a dictionary which maps task_number to the tuple containing currently saved file name and validation accuracy
        ie. {task_number: (file_name, validation_accuracy)}
        """
        self.ckpt_dir = path
        self.LOG = LOG
        self.tasks = {}

        if self.ckpt_dir[-1] != '/':
            self.ckpt_dir = self.ckpt_dir+'/'

    def remove_old_save(self, task_number):
        if task_number in self.tasks.key():
            task_model_path = os.path.join(os.path.join(self.ckpt_dir, str(task_number)), self.tasks[task_number][0])
            self.LOG.info("Removing Old Model Checkpoint at " + task_model_path)
            os.remove(task_model_path)

    def create_new_save(self, name, net, task_number):
        self.LOG.info("Saving New Best Validation Accuracy: "+str(self.best_acc))
        task_model_path = os.path.join(os.path.join(self.ckpt_dir, str(task_number)), name)
        self.tasks[task_number][1] = name
        self.remove_old_save()
        self.LOG.info("Creating New Model Checkpoint at " + str(task_model_path))
        torch.save(net, task_model_path)

    def update(self, new_acc, task_number):
        if task_number not in self.tasks:
            old_acc = 0
            self.tasks[task_number] = (0, "No_model.pth")
        else:
            old_acc = self.tasks[task_number][1]
        better = new_acc > old_acc
        if better:
            self.tasks[task_number][0] = new_acc
        return better


