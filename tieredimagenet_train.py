#Unofficial Implementation of MAML (TieredImageNet)

import torch
import numpy as np
import argparse
import learn2learn.data as data

from torchvision import transforms
from Tieredimagenet import TieredImagenet as Tiered
from torch.utils.data import DataLoader
from conv_model_architecture import Conv_block
from meta import Meta
from learn2learn.data.transforms import (NWays, KShots, LoadData, RemapLabels)
from learn2learn.data.utils import partition_task

torch.manual_seed(45)
torch.cuda.manual_seed_all(430)
np.random.seed(100)
device = torch.device('cuda')

def main():

    #Initialized Backbone Architecture (-> conv4)
    network = Conv_block(args.imgc, args.n_way, args.num_filters).to(device) 

    #MAML algorithm
    maml = Meta(args, network).to(device)
  
    #Load Tiered-Imagenet via learn2learn package/library (borrowed from learn2learn)
    train = Tiered(args.datasets_root, transform=transforms.ToTensor(), mode="train", download=True)
    val = Tiered(args.datasets_root, transform=transforms.ToTensor(), mode="validation", download=True)
    
    train_dataset = data.MetaDataset(train)
    val_dataset = data.MetaDataset(val)
    
    train_transforms = [
        NWays(train_dataset, n = args.n_way),
        KShots(train_dataset, k =  args.k_qry + args.k_spt),
        LoadData(train_dataset),
        RemapLabels(train_dataset)
        ]
    val_transforms = [
        NWays(val_dataset, args.n_way),
        KShots(val_dataset, args.k_qry*2),
        LoadData(val_dataset),
        RemapLabels(val_dataset)
        ]
    
    
    train_tasks = data.TaskDataset(train_dataset, task_transforms = train_transforms, num_tasks = args.epochs*args.task_num)
    val_tasks = data.TaskDataset(val_dataset, task_transforms = val_transforms, num_tasks=args.val_task)
    
    train_loader = DataLoader(train_tasks, batch_size = args.task_num, pin_memory=True, shuffle = True)
    val_loader = DataLoader(val_tasks, pin_memory=True, shuffle = True)

    best_acc = 0

    #Train MAML
    for epoch in range(args.epochs):
        
        #few-shot setting
        x_spt_, y_spt_, x_qry_, y_qry_ = [], [], [], []
        x, y = next(iter(train_loader))
        for i in range(args.task_num):
            (x_spt, y_spt), (x_qry, y_qry) = partition_task(x[i].to(device), y[i].to(device), shots=args.k_spt)
            x_spt_.append(x_spt), y_spt_.append(y_spt), x_qry_.append(x_qry), y_qry_.append(y_qry)

        result = maml(x_spt_, y_spt_, x_qry_, y_qry_) 
        
        ## Print the result at every 100 epochs
        if (epoch+1) % 100 == 0 or epoch == 0:
            print_result = 'epoch: {0} \ttraining acc: {1:0.4f} \tloss: {2:0.4f}'.format(epoch+1, result[0], result[1])
            print(print_result)
            
        # Evaluate at every 500 epochs (Meta-Validation)
        if (epoch+1) % 500 == 0:
            accs_all_test = []
            all_loss = []
            for _ in range(args.val_task):
                x_val, y_val = next(iter(val_loader))
                (x_spt_val, y_spt_val), (x_qry_val, y_qry_val) = partition_task(x_val.squeeze(0).to(device), y_val.squeeze(0).to(device), shots=args.k_spt)

                result_test = maml.validation(x_spt_val, y_spt_val, x_qry_val, y_qry_val)
                accs_all_test.append(result_test[0])
                all_loss.append(result_test[1])
                
            accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            loss = np.array(all_loss).mean(axis=0).astype(np.float16)

            print_result_val =  'epoch: {0} \tvalidation acc: {1:0.4f} \tloss: {2:0.4f} **'.format(epoch+1, accs, loss)
            print(print_result_val)
            #save the best model (result from validation datasets)
            if best_acc <= accs:
                best_acc = accs
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': result[2].state_dict(),
                            'loss': result[1]
                            },"/data01/jjlee_hdd/save_model/final_model/Tiered_5-"+str(args.k_spt)+"_"+str(args.version)+".pt" )
            else: 
                best_acc = best_acc 
            print("best_val_acc: {0:0.4f}".format(best_acc))
        
        if epoch == (args.epochs - 1) :
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': result[2].state_dict(),
                        'loss': result[1]
                        },"/data01/jjlee_hdd/save_model/final_model/Tiered_5-"+str(args.k_spt)+"_"+str(args.version)+".pt" )   

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--num_filters', type=int, help='size of filters of convblock', default=32)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--val_task', type=int, help='number of tasks for validation', default=600)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument("--version", type=int, help='version of MAML', default=0)
    argparser.add_argument("--datasets_root", type=str, help='version of MAML', default='/data01/jjlee_hdd/dataset_tieredimagenet/')
    
    args = argparser.parse_args()

    main()
