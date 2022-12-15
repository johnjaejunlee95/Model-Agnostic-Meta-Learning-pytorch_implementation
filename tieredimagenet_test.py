#Unofficial Implementation of MAML (miniImageNet)

import torch
import numpy as np
import argparse
import learn2learn as l2l

from torchvision import transforms
from Tieredimagenet import TieredImagenet as Tiered
from torch.utils.data import DataLoader
from conv_model_architecture import Conv_block
from meta import Meta
from learn2learn.data.transforms import (NWays, KShots, LoadData, RemapLabels)
from learn2learn.data.utils import  partition_task

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
device = torch.device('cuda')

def main():
    
    network = Conv_block(args.imgc, args.n_way, args.num_filters).to(device)
    checkpoint = torch.load("/data01/jjlee_hdd/save_model/final_model/Tiered_5-"+str(args.k_spt)+"_"+str(args.version)+".pth")
    network.load_state_dict(checkpoint['model_state_dict'])
    
    maml = Meta(args, network).to(device)

    test = Tiered(args.datasets_root, transform=transforms.ToTensor(), mode="test")
    test_dataset = l2l.data.MetaDataset(test)
        
    test_transforms = [
        NWays(test_dataset, n = args.n_way),
        KShots(test_dataset, k = args.k_spt*2),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
    ]

    test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms = test_transforms, num_tasks=args.test_num_task)
    test_loader = DataLoader(test_tasks, pin_memory=True, shuffle = True)
    
    accs_all_test = []
    all_loss = []
    
    for _ in range(args.test_num_task):
        x_test, y_test = next(iter(test_loader))
        (x_spt, y_spt), (x_qry, y_qry) = partition_task(x_test.squeeze(0).to(device), y_test.squeeze(0).to(device), shots=args.k_spt)

        model_test = maml.validation(x_spt, y_spt, x_qry, y_qry)
        accs_all_test.append(model_test[0])
        all_loss.append(model_test[1])
    
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
    loss = np.array(all_loss).mean(axis=0).astype(np.float16)
    stds = np.std(np.array(accs_all_test), 0)
    ci95 = 1.96*stds/np.sqrt(args.test_num_task)
    
    print(np.around(accs*100, 2), np.around(loss, 2), np.around(ci95*100, 2))
    
if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--num_filters', type=int, help='size of filters of convblock', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--test_num_task', type=int, help='number of tasks for validation', default=600)
    argparser.add_argument("--version", type=int, help='version of MAML', default=0)
    argparser.add_argument("--datasets_root", type=str, help='version of MAML', default='/data01/jjlee_hdd/dataset_tieredimagenet/')
    
    args = argparser.parse_args()
    
    main()
