import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import subprocess
import torch
import torchvision
from torchvision import transforms
import tqdm
from pdb import set_trace as st
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

path = os.path.join('./dataset/', 'tiny-imagenet-200')

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def create_dir(dirpath, print_description=""):
    # Function to create directory
    # Checks if path exists
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
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]), 
                 target_transform=None, subset=None):
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
    file_path = os.path.join(path, "wnids.txt")
    
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

def file_reading_func(root="/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt"):
    # Reads an annotation file and sorts it in the form of dictionary
    # file_name -> classname 
    f = open(root, "r")
    out_dict = {}
    for x in f:
        cur_line = x.rstrip('\n').split('\t')
        out_dict[cur_line[0]] = cur_line[1]
    return out_dict

def traverse(root):
    # Helper function prints whatever is in this root
    for path, currentDirectory, files in os.walk(root):
        for file in files:
            print(os.path.join(path, file))
    
        
def reorganize_files(new_root="/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200/new_test", 
                     old_root="/root/yifei/CL_mod/src/dataset/tiny-imagenet-200/tiny-imagenet-200/val", 
                     file_reading_func=file_reading_func):
    # Function that creates a new folder for imgfolder format, used for test set in tiny imageNet
    # new_root should be an empty path where the new folder is created
    # old_root is where the annotated txt file and the images are stored
    # file_reading_func processes
    # val/classname/*.jpeg
    # To save space, does not copy but moves the files 
    # Creates new images in the format *class_name*_*sample_number*.JPEG
    ann_root = os.path.join(old_root, "val_annotations.txt")
    file_dict = file_reading_func(ann_root)
    create_dir(new_root)
    file_count = {}
    for path, currentDirectory, files in os.walk(old_root):
        for file in files:
            if file.split(".")[-1] == "JPEG":
                # Only move JPEG
                current_path = os.path.join(path, file)
                class_name = file_dict[file]
                try:
                    current_count = file_count[class_name]
                except:
                    file_count[class_name] = 0
                    current_count = 0
                    create_dir(os.path.join(new_root, class_name))
                new_path = os.path.join(new_root, class_name)
                new_img_name = str(class_name)+"_"+str(current_count)+".JPEG"
                new_img_path = os.path.join(new_path, new_img_name)
                file_count[class_name] += 1
                shutil.move(current_path, new_img_path)
    print("Finished processing all images")
            
            
            
    

    
    
    
    
    
    
    
    
    