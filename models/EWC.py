import math
import os
import time
from collections import Counter
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import nn

### Code modified from https://github.com/Mattdl/CLsurvey

def per_task_updates(net, dataset_dict, writer, LOG, task_info, args, task_number, use_cuda=torch.cuda.is_available(), head_shared = False, reg_lambda=1):
    
    # compute fisher if not first task
    if 'prev_train_dataset' in dataset_dict.keys():
        prev_dataset = dataset_dict['prev_train_dataset']
        # store previous omega values (Accumulated Fisher)
        reg_params = store_prev_reg_params(net)
        net.reg_params = reg_params
        data_len = len(prev_dataset.dataset)
        model_ft = diag_fisher(net, prev_dataset, data_len)
        # accumulate the current fisher with the previosly computed one
        reg_params = accumelate_reg_params(net)
        model_ft.reg_params = reg_params
        
    else:
        reg_params = initialize_reg_params(net)
        net.reg_params = reg_params
    
    dset_classes = [task_info[task_number][i] for i in task_info[task_number]]
    print("Current class labels: "+str(dset_classes))
    # print current omega stat.
    sanitycheck(net, printing=task_number) # Do not print first run 
    net.reg_params['lambda'] = reg_lambda
    
    # get the number of features in this network and add a new task head
    if not head_shared:
        try:
            num_ftrs = net.fc.out_features
            new_fc = nn.Sequential(
                net.fc,
                nn.ReLU(),
                nn.Linear(num_ftrs, len(dset_classes))
            )
            net.fc = new_fc
        except:
            last_layer_index = str(len(net.fc._modules) - 1)
            num_ftrs = net.fc._modules[last_layer_index].in_features
            net.fc._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
        print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))

    criterion = nn.CrossEntropyLoss()
    # update the objective based params
    if use_cuda:
        net = net.cuda()

    # call the EWC optimizer
    optimizer = Weight_Regularized_SGD(net.parameters(), lr=args.lr, momentum=args.mt, weight_decay=args.wd, use_cuda=use_cuda)

    # train the model
    # this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft, acc = train_model(net, criterion, optimizer, args.lr, dataset_dict, use_cuda,
                                args.num_epochs, logger=LOG, writer=writer,
                                exp_dir=os.path.join(args.checkpoint_dir, str(task_number)), task_number=task_number)

    return model_ft, 
    
# def train_model()


def train_model(model, criterion, optimizer, lr, dset_loaders, use_gpu, num_epochs, logger, writer, task_number, exp_dir='./',
                resume='', saving_freq=5):
    """
    Trains a deep learning model with EWC penalty.
    Empirical Fisher is used instead of true FIM for efficiency and as there are no significant differences in results.
    :return: last model & best validation accuracy
    """
    since = time.time()
    if os.path.isfile(resume):
        logger.info("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        val_beat_counts = checkpoint['val_beat_counts']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = checkpoint['lr']
        logger.info("lr is ", lr)
        logger.info("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        logger.info("=> no checkpoint found at '{}'".format(resume))
        val_beat_counts = 0
        best_acc = 0

    logger.info(str(start_epoch))
    logger.info("lr is "+str(lr))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    logger.info('Best val Acc: {:4f}'.format(best_acc))
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                # FOR MNIST DATASET
                inputs = inputs.squeeze()

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # call the optimizer and pass reg_params to be utilized in the EWC penalty
                    optimizer.step(model.reg_params)

                # statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dset_loaders[phase].dataset)
            epoch_acc = running_corrects / len(dset_loaders[phase].dataset)

            writer.add_scalar(str(task_number)+'_'+str(phase)+' Loss', epoch_loss, epoch)
            writer.add_scalar(str(task_number)+'_'+str(phase)+' Acc', epoch_acc, epoch)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if math.isnan(epoch_loss):
                return model, best_acc
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    logger.info("FOUND NEW BEST VAL MODEL: "+str(epoch_acc)+" on epoch "+str(epoch))
                    del outputs, labels, inputs, loss, preds
                    best_acc = epoch_acc
                    epoch_file_name = exp_dir + '/' + 'epoch' + '_best_model.pth.tar'
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'lr': lr,
                        'val_beat_counts': val_beat_counts,
                        'epoch_acc': epoch_acc,
                        'best_acc': best_acc,
                        'arch': 'alexnet',
                        'model': model,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, epoch_file_name)
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
        if epoch % saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)

    time_elapsed = time.time() - since
    logger.info('Training task complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc



class Weight_Regularized_SGD(optim.SGD):
    r"""Implements stochastic gradient descent with an EWC penalty on important weights for previous tasks
    Code modified from https://github.com/Mattdl/CLsurvey
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, use_cuda=False):
        super(Weight_Regularized_SGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.use_cuda = use_cuda

    def __setstate__(self, state):
        super(Weight_Regularized_SGD, self).__setstate__(state)

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            reg_params: a dictionary where importance weights for each parameter is stored.
        """

        loss = None
        if closure is not None:
            loss = closure()
        index = 0
        reg_lambda = reg_params.get('lambda')  # a hyper parameter for the EWC regularizer

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # This part is to add the gradients of the EWC regularizer

                if p in reg_params:
                    # for each parameter considered in the optimization process
                    reg_param = reg_params.get(
                        p)  # get the corresponding dictionary where the information for EWC penalty is stored
                    omega = reg_param.get('omega')  # the accumelated Fisher information matrix.
                    init_val = reg_param.get('init_val')  # theta*, the optimal parameters up until the previous task.
                    curr_wegiht_val = p.data  # get the current weight value
                    # move the variables to cuda
                    if self.use_cuda:
                        init_val = init_val.cuda()
                        omega = omega.cuda()

                    # get the difference
                    weight_dif = curr_wegiht_val.add(-1, init_val)  # compute the difference between theta and theta*,

                    regulizer = weight_dif.mul(2 * reg_lambda * omega)  # the gradient of the EWC penalty
                    d_p.add_(regulizer)  # add the gradient of the penalty

                    # delete unused variables
                    del weight_dif, curr_wegiht_val, omega, init_val, regulizer
                # The EWC regularizer ends here
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)
                index += 1
        return loss
    
    
def initialize_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
    reg_params = {}
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            print('initializing param', name)
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            # initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
    return reg_params


def store_prev_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('storing previous omega', name)
                prev_omega = reg_param.get('omega')
                new_omega = torch.FloatTensor(param.size()).zero_()
                init_val = param.data.clone()
                reg_param['prev_omega'] = prev_omega
                reg_param['omega'] = new_omega

                # initialize the initial value to that before starting training
                reg_param['init_val'] = init_val
                reg_params[param] = reg_param

        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params

def diag_fisher(model, dset_loader, data_len):
    reg_params = model.reg_params
    model.eval()

    for data in dset_loader:
        model.zero_grad()
        x, label = data
        x, label = Variable(x).cuda(), Variable(label, requires_grad=False).cuda()

        output = model(x)
        loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label, size_average=False)
        loss.backward()

        for n, p in model.named_parameters():
            if p in reg_params:
                reg_param = reg_params.get(p)
                omega = reg_param['omega'].cuda()
                omega += p.grad.data ** 2 / data_len  # Each datasample only contributes 1/datalength to the total
                reg_param['omega'] = omega
    return model

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def accumelate_reg_params(model, freeze_layers=None):
    freeze_layers = [] if freeze_layers is None else freeze_layers
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        if not name in freeze_layers:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('restoring previous omega', name)
                prev_omega = reg_param.get('prev_omega')
                prev_omega = prev_omega.cuda()

                new_omega = (reg_param.get('omega')).cuda()
                acc_omega = torch.add(prev_omega, new_omega)

                del reg_param['prev_omega']
                reg_param['omega'] = acc_omega

                reg_params[param] = reg_param
                del acc_omega
                del new_omega
                del prev_omega
        else:
            if param in reg_params:
                reg_param = reg_params.get(param)
                print('removing unused omega', name)
                del reg_param['omega']
                del reg_params[param]
    return reg_params

def sanitycheck(model, printing=False):
    for name, param in model.named_parameters():
        if param in model.reg_params:
            reg_param = model.reg_params.get(param)
            omega = reg_param.get('omega')
            if printing:
                print('omega max is', omega.max().item())
                print('omega min is', omega.min().item())
                print('omega mean is', omega.mean().item())  # here the ewc code goes
                
                
def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count > 10:
        continue_training = False
        print("training terminated")
    if count == 5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training
