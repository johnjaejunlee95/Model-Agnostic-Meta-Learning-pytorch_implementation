import torch
import numpy as np

from collections import OrderedDict
from torch import nn
from torch import optim
from torch.nn import functional as F
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Meta-Learning (Inner-Loop & Outer Loop)
class Meta(nn.Module):

    def __init__(self, args, network):

        super(Meta, self ).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.network = network
        self.loss = nn.CrossEntropyLoss()
        self.meta_optim = optim.Adam(self.network.parameters(), lr=self.meta_lr)
        
    
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        
        task_num = len(x_spt)
        querysz = x_qry[0].size(0)
        losses_q = torch.tensor(0.0, requires_grad=True)
        corrects = 0

        ### Inner Loop
        self.network.train()
        for i in range(task_num): # Iterate the # of batch-size (-> 4)
            weights = OrderedDict((key, params) for (key, params) in self.network.named_parameters())

            for _ in range(self.update_step): # Iterate the # of Inner-Loop steps (training setting -> 5)
                
                logits = self.network(x_spt[i], weights) 
                loss_ = self.loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss_, weights.values(), create_graph=True)
                weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(weights.items(), grad))
                    
            logits_q = self.network(x_qry[i], weights)
            loss_q = self.loss(logits_q, y_qry[i])
            
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
            
            corrects += correct
            losses_q = losses_q + loss_q
        
        ### Outer Loop
        overall_loss = losses_q / task_num # Average Loss
        
        self.meta_optim.zero_grad()
        overall_loss.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num) # Accuracy

        return accs, overall_loss.detach().cpu().numpy(), self.network

    def validation(self, x_spt, y_spt, x_qry, y_qry):
        
        querysz = x_qry.size(0)
        network = deepcopy(self.network) # copy the meta-parameters not to update during Validation/Test
        
        weights = OrderedDict((key, params) for (key, params) in network.named_parameters())

        for _ in range(self.update_step_test): # Iterate the # of Inner-Loop steps (validation/test gradient steps -> 10)
            
            logits = network(x_spt, weights)
            loss_ = self.loss(logits, y_spt)
            grad = torch.autograd.grad(loss_, weights.values(), create_graph=True)
            weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(weights.items(), grad))

        with torch.no_grad():
            logits_q = network(x_qry, weights)
            loss_q = self.loss(logits_q, y_qry)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item() 
        
        accs = np.array(correct) / querysz # Accuracy

        return accs, loss_q.detach().cpu().numpy()
    
def main():
    pass

if __name__ == '__main__':
    main()
