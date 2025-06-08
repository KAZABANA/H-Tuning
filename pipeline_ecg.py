# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:25:30 2023

@author: COCHE User
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from datacollection import ECGdataset_prepare_finetuning_sepe
from model_src_ecg.model_code_default import NN_default, Cutmix,Cutmix_student
from pytorchtools import EarlyStopping
from evaluation import print_result, find_thresholds
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from Half_Trainer import HalfTrainer
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## model validation on single GPU
def validate(model, valloader, device, iftest=False, threshold=0.5 * np.ones(5), iftrain=False, args=None):
    model.eval()
    losses, probs, lbls, logit = [], [], [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
        with torch.no_grad():
            out = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    else:
        threshold = find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result

def mask_ecg_signal(signal, valid_lead_num):
    if valid_lead_num == 1:
        mask_lead = np.arange(1,12)
        signal[:, mask_lead, :, :] = 0
        return signal
    elif valid_lead_num == 3:
        mask_lead = [0, 2, 3, 4, 5, 7, 8, 9, 11] #[1,6,10],II, V1, V5
        signal[:, mask_lead, :, :] = 0
        return signal
    else:
        return signal

def validate_student(model, valloader, device, iftest=False, threshold=0.5 * np.ones(5), iftrain=False, args=None):
    model.eval()
    losses, probs, lbls, logit = [], [], [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
        with torch.no_grad():
            inp_windows_t = mask_ecg_signal(inp_windows_t, args.leads_for_student)
            #print(inp_windows_t[0, :, 0, :])
            out = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    else:
        threshold = find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result

def count_parameters(model):
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'bias' not in n and n !='classifier.1.weight':
            # print(n)
            p.requires_grad = False
    return

def load_pretrained_model(net, path, args, device='cuda:0'):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.find('classifier.1') < 0}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def model_config_initialization(args, loader_train, device='cuda:0'):
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    file_name_pretrain = args.pretrain_dataset + model_config + 'bias_full_checkpoint.pkl'
    print(file_name_pretrain)
    setup_seed(args.seed)
    r = args.r#
    num_layers, complexity = 47, 512
    num_leads = 12
    num_class = args.num_class
    print('current method double check:', args.ranklist)
    print('current rank double check:', r)
    if args.ranklist == 'lora_ave': ## equals LoRA
        net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                         rank_list=r)
        mark_only_lora_as_trainable(net)
    else: ## equals Full fine-tuning
        net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,rank_list=0)
    net.to(device)
    net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
    print(path + file_name_pretrain)
    if 'lora' in args.ranklist:
        params_to_update = []
        for name, param in net.named_parameters():
            if name.find('lora') > -1:
                params_to_update.append(param)
            elif name.find('bias') > -1:
                params_to_update.append(param)
            elif name.find('classifier.1.weight') > -1:
                params_to_update.append(param)
        optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=0.001)
    return net, optimizer

def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def loading_lora_checkpoint(net, path,device='cuda:0'):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load(path))
    return net
def pipeline_start_default(args): ## pipeline for model fine-tuning on downstream datasets
    print('semiconfig:', args.semi_config)
    print(args.ranklist)
    device = args.device
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = 128
    print('learning rate:', args.learning_rate)
    print('batch_size:', batch_size)
    print('interval', args.interval)
    setup_seed(args.seed)
    dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight = ECGdataset_prepare_finetuning_sepe(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.finetune_epoch
    if args.finetune_dataset == 'WFDB_Ga' or args.finetune_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2
    print('max_iteration:', iteration)
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
        args.finetune_label_ratio) + 'seed' + str(args.seed)
    print('save_name:',save_name)
    start_time = time.time()
    net, optimizer = model_config_initialization(args, loader_train, device=device)
    if args.zo_config == 'FT_cls':
        print('linear probing')
        for n, p in net.named_parameters():
            if 'class' not in n:
                p.requires_grad = False
            else:
                print(n)
    early_stopping = EarlyStopping(10, verbose=True,dataset_name=save_name,delta=0, args=args)  # 15
    step = 0
    net.train()
    setup_seed(args.seed)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    running_loss = 0.0
    if 'FT' not in args.zo_config:
        print('Extra Trainer is required')
        Trainer = HalfTrainer(optimizer, my_lr_scheduler, None, args)
    else:
        print('Extra Trainer is not required')
    #count_parameters(net)
    for current in range(iteration):
        if current % args.interval == 0:
            print('training_loss:', running_loss/args.interval)
            running_loss = 0.0
            valid_result = validate(net, loader_valid, device)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        ## mini-batch sampling
        try:
            images, labels = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            images, labels = Cutmix(images, labels, device)
        net.train()
        optimizer.zero_grad()
        outputs = net(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
        my_lr_scheduler.step()
        running_loss += loss.item()
        step += 1
    end_time = time.time()
    running_time = (end_time - start_time) / (current + 1)
    print(f"running time {running_time:.2f} 秒")
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
    print('load_name:', save_name + '_checkpoint.pkl')
    net.load_state_dict(torch.load(path + save_name + '_checkpoint.pkl', map_location=device))
    trainable_num = count_parameters(net)
    print('trainable_num:', trainable_num)
    net.eval()
    with torch.no_grad():
        valid_result = validate(net, loader_valid, device)
        test_result = validate(net, loader_test, device, iftest=True, threshold=valid_result['threshold'])
        test_result.update({'trainable_num': trainable_num})
        test_result.update({'memory': allocated_memory})
        test_result.update({'time': running_time})
    return test_result,net

def kd_teacher_learning_ho(args):
    args.ranklist = 'lora_ave'
    print('ranklist:', args.ranklist)
    device = args.device
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = args.batch_size
    if batch_size > 64:
        args.learning_rate = 0.002
    else:
        args.learning_rate = 0.001
    print('learning rate:', args.learning_rate)
    print('batch_size:', batch_size)
    print('interval',args.interval)
    setup_seed(args.seed)
    dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight = ECGdataset_prepare_finetuning_sepe(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.finetune_epoch
    if args.finetune_dataset == 'WFDB_Ga' or args.finetune_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2
    print('max_iteration:', iteration)
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
        args.finetune_label_ratio) + 'seed' + str(args.seed)
    print('save_name:',save_name)
    start_time = time.time()
    net, optimizer = model_config_initialization(args, loader_train, device=device)
    early_stopping = EarlyStopping(10, verbose=True,dataset_name=save_name,delta=0, args=args)  # 15
    step = 0
    net.train()
    setup_seed(args.seed)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    running_loss = 0.0
    Trainer = HalfTrainer(optimizer, my_lr_scheduler, None, args)
    for current in range(iteration):
        if current % args.interval == 0:
            print('training_loss:', running_loss/args.interval)
            running_loss = 0.0
            valid_result = validate(net, loader_valid, device)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        ## mini-batch sampling
        try:
            images, labels = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            images, labels = Cutmix(images, labels, device)
        net.train()
        optimizer.zero_grad()
        if args.tune_bp:
            Trainer.tune_with_pure_bp(net, images, labels)
        else:
            Trainer.ho_step(net, images, labels)
        step += 1
    end_time = time.time()
    running_time = (end_time - start_time) / (current + 1)
    print(f"running time {running_time:.2f} 秒")
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
    print('load_name:', save_name + '_checkpoint.pkl')
    net = loading_lora_checkpoint(net, path + save_name + '_checkpoint.pkl', device=args.device)
    trainable_num = count_parameters(net)
    print('trainable_num:', trainable_num)
    net.eval()
    print('merging...')
    net.merge_net()
    with torch.no_grad():
        valid_result = validate(net, loader_valid, device)
        test_result = validate(net, loader_test, device, iftest=True, threshold=valid_result['threshold'])
        test_result.update({'trainable_num': trainable_num})
        test_result.update({'memory': allocated_memory})
        test_result.update({'time': running_time})
    return test_result, net

def kd_student_learning_ho(args):
    FT_teacher_list=['FT']
    if args.zo_config in FT_teacher_list:
        print(args.zo_config)
        print('We adopt full tuning on teacher')
        args.ranklist = 'FT'
        teacher_results, teacher_net = pipeline_start_default(args)
    else:
        print('We adopt LoRA tuning on teacher')
        teacher_results, teacher_net = kd_teacher_learning_ho(args)
    teacher_net.eval()
    args.ranklist = 'FT'
    print('semiconfig:', args.semi_config)
    device = args.device
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = 128
    args.learning_rate = 0.002
    print('learning rate:', args.learning_rate)
    print('batch_size:', batch_size)
    print('interval',args.interval)
    setup_seed(args.seed)
    dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight = ECGdataset_prepare_finetuning_sepe(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.finetune_epoch
    if args.finetune_dataset == 'WFDB_Ga' or args.finetune_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2
    print('max_iteration:', iteration)
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
        args.finetune_label_ratio) + 'seed' + str(args.seed)
    print('save_name:',save_name)
    start_time = time.time()
    num_layers, complexity = 14, 64
    net = NN_default(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=0)
    if args.student_load_pretrain:
        file_name_pretrain = args.pretrain_dataset + 'tinylight' + 'bias_full_checkpoint.pkl'
        print('loading student pretrain')
        net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    net.to(device)
    early_stopping = EarlyStopping(10, verbose=True,dataset_name=save_name,delta=0, args=args)  # 15
    step = 0
    net.train()
    setup_seed(args.seed)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    running_loss = 0.0
    for current in range(iteration):
        if current % args.interval == 0:
            print('training_loss:', running_loss/args.interval)
            running_loss = 0.0
            valid_result = validate_student(net, loader_valid, device, args = args)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        ## mini-batch sampling
        try:
            images, labels = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            images, labels = Cutmix(images, labels, device)
            with torch.inference_mode():
                teacher_outputs = teacher_net(images)
                images = mask_ecg_signal(images, args.leads_for_student)
        net.train()
        optimizer.zero_grad()
        inputs = images
        outputs = net(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss += F.binary_cross_entropy_with_logits(outputs, teacher_outputs.sigmoid())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        my_lr_scheduler.step()
    end_time = time.time()
    running_time = (end_time - start_time) / (current + 1)
    print(f"running time {running_time:.2f} 秒")
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
    print('load_name:', save_name + '_checkpoint.pkl')
    net.load_state_dict(torch.load(path + save_name + '_checkpoint.pkl', map_location=device))
    trainable_num = count_parameters(net)
    print('trainable_num:', trainable_num)
    net.eval()
    with torch.no_grad():
        valid_result = validate_student(net, loader_valid, device, args = args)
        test_result = validate_student(net, loader_test, device, iftest=True, threshold=valid_result['threshold'], args = args)
        test_result.update({'trainable_num': trainable_num})
        test_result.update({'memory': allocated_memory})
        test_result.update({'time': running_time})
        test_result.update({'teacher_result': teacher_results})
    return test_result

def student_learning_without_tea(args): ## student learning without teacher guidance.
    args.ranklist = 'FT'
    print('semiconfig:', args.semi_config)
    device = args.device
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    batch_size = 128
    args.learning_rate = 0.002
    print('learning rate:', args.learning_rate)
    print('batch_size:', batch_size)
    print('interval',args.interval)
    setup_seed(args.seed)
    dataset_train, dataset_valid, dataset_test, positive_weight, negative_weight = ECGdataset_prepare_finetuning_sepe(args=args)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    label_iter = iter(loader_train)
    iteration = len(loader_train) * args.finetune_epoch
    if args.finetune_dataset == 'WFDB_Ga' or args.finetune_dataset == 'WFDB_ChapmanShaoxing':
        iteration = iteration * 2
    print('max_iteration:', iteration)
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
        args.finetune_label_ratio) + 'seed' + str(args.seed)
    print('save_name:',save_name)
    start_time = time.time()
    num_layers, complexity = 14, 64
    net = NN_default(nOUT=args.num_class, complexity=complexity, inputchannel=12, num_layers=num_layers, rank_list=0)
    if args.student_load_pretrain:
        file_name_pretrain = args.pretrain_dataset + 'tinylight' + 'bias_full_checkpoint.pkl'
        print('loading student pretrain')
        net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    net.to(device)
    early_stopping = EarlyStopping(10, verbose=True,dataset_name=save_name,delta=0, args=args)  # 15
    step = 0
    net.train()
    setup_seed(args.seed)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    running_loss = 0.0
    for current in range(iteration):
        if current % args.interval == 0:
            print('training_loss:', running_loss/args.interval)
            running_loss = 0.0
            valid_result = validate_student(net, loader_valid, device, args = args)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        ## mini-batch sampling
        try:
            images, labels = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        images = images.float().to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            if args.leads_for_student == 12:
                images, labels = Cutmix(images, labels, device)
            else:
                images, labels = Cutmix_student(images, labels, device, valid_lead_num=args.leads_for_student)
                images = mask_ecg_signal(images, args.leads_for_student)
        net.train()
        optimizer.zero_grad()
        inputs = images
        outputs = net(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        # print(net.classifier[1].weight.shape)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        my_lr_scheduler.step()
    end_time = time.time()
    running_time = (end_time - start_time) / (current + 1)
    print(f"running time {running_time:.2f} 秒")
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
    print('load_name:', save_name + '_checkpoint.pkl')
    net.load_state_dict(torch.load(path + save_name + '_checkpoint.pkl', map_location=device))
    trainable_num = count_parameters(net)
    print('trainable_num:', trainable_num)
    net.eval()
    with torch.no_grad():
        valid_result = validate_student(net, loader_valid, device, args=args)
        test_result = validate_student(net, loader_test, device, iftest=True, threshold=valid_result['threshold'],
                                       args=args)
        test_result.update({'trainable_num': trainable_num})
        test_result.update({'memory': allocated_memory})
        test_result.update({'time': running_time})
    return test_result


