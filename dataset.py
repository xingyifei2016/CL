import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import subprocess
import torch


from pdb import set_trace as st

path = os.path.join('./dataset/', 'tiny-imagenet-200')

# Function to create directory
def create_dir(dirpath, print_description=""):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, mode=0o750)
    except Exception as e:
        print(e)
        print("ERROR IN CREATING ", print_description, " PATH:", dirpath)
        
def download_tinyImagenet(path):
    create_dir(path)
    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200.zip')):
        subprocess.call(
            "wget -P {} http://cs231n.stanford.edu/tiny-imagenet-200.zip".format(path),
            shell=True)
        print("Succesfully downloaded TinyImagenet dataset.")
    else:
        print("Already downloaded TinyImgnet dataset in {}".format(path))
    if not os.path.exists(os.path.join(path, 'tiny-imagenet-200')):
        subprocess.call(
            "unzip {} -d {}".format(os.path.join(path, 'tiny-imagenet-200.zip'), path),
            shell=True)
        print("Succesfully extracted TinyImagenet dataset.")
    else:
        print("Already extracted TinyImgnet dataset in {}".format(os.path.join(path, 'tiny-imagenet-200')))

class tinyImageNet(torchvision.datasets.ImageFolder):
    # Class that inherits from imagefolder for dataloading purposes
    # Applies basic transforms and accepts a subset of classes for training/testing
    # Modifies the find_classes function from torch's Dataset class
    # Remember in defying paper, train is split 80:20 to train/val; val is for testing
    def __init__(self, root="/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200/train", 
                 transform=None, target_transform=None, subset=None):
        # Subset stores a dictionary of class names to its label
        self.subset = subset
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        self.all_classes = torchvision.datasets.folder.find_classes(root)
        
    def find_classes(self, path):
        if self.subset:
        # If using a subset of all classes, then only use those classes
            return self.subset.keys(), self.subset 
        else:
        # Else, use torch's generalized class function
            return torchvision.datasets.folder.find_classes(path)
    
def create_tasks(num_tasks=10, path="/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200"):
    # Splitting into n classes creates a n-length array, where each item in the array consists of
    # a dictionary that maps class_labels to class_id (ie 0)
     
    # Get the file path that stores the class names
    file_path = os.path.join(root_path, "wnids.txt")
    
    # Get the class names as a list
    lines = [line.rstrip('\n') for line in open(file_path)]
    
    # Make sure the tasks are correctly assigned
    nb_classes_task = len(lines) // num_tasks
    print("Split "+str(len(lines))+" classes into "+ str(num_tasks)+" tasks with "+str(nb_classes_task)+" classes per task")
    assert len(lines) % num_tasks == 0, "total "+str(len(lines))+" classes must be divisible by nb classes per task"
    
    # Create a dictionary to return in this format: class_labels -> class_id
    outputs = []
    current_id = 0
    for i in range(num_tasks):
        class_lbl = lines[i*num_tasks:(i+1)*num_tasks]  
        task_dict = {} 
        for j in range(i*num_tasks, (i+1)*num_tasks):
            task_dict[lines[j]] = current_id
            current_id += 1
        outputs.append(task_dict) 
    return outputs
 
    


        
        
    
    
    
    
    
    