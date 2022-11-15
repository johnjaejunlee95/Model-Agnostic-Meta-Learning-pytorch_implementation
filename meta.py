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

    def __init__(self, args):

        super(Meta, self ).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x_spt, y_spt, x_qry, y_qry, network):

        meta_optim = optim.Adam(network.parameters(), lr=self.meta_lr) # meta-update optimizer
        
        task_num = len(x_spt)
        querysz = x_qry[0].size(0)
        losses_q = torch.tensor(0.0, requires_grad=True)
        corrects = 0

        ### Inner Loop
        network.train()
        for i in range(task_num): # Iterate the # of batch-size (-> 4)
            weights = OrderedDict((key, params) for (key, params) in network.named_parameters())

            for k in range(self.update_step): # Iterate the # of Inner-Loop steps (training setting -> 5)
                
                if k == 0:
                    logits = network(x_spt[i], weights) 
                    loss_ = self.loss(logits, y_spt[i])
                    grad = torch.autograd.grad(loss_, weights.values(), create_graph=True, allow_unused=True)
                    torch.nn.utils.clip_grad_norm_(weights.values(), 5)
                    updated_weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(weights.items(), grad))
                else:
                    logits = network(x_spt[i], updated_weights)
                    loss_ = self.loss(logits, y_spt[i])
                    grad = torch.autograd.grad(loss_, updated_weights.values(), create_graph=True, allow_unused=True)
                    torch.nn.utils.clip_grad_norm_(updated_weights.values(), 5)
                    updated_weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(updated_weights.items(), grad))
                
                if k == self.update_step - 1 :
                    weights = updated_weights
                    
            logits_q = network(x_qry[i], weights)
            loss_q = self.loss(logits_q, y_qry[i])
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects += correct
            losses_q = losses_q + loss_q
        
        ### Outer Loop
        overall_loss = losses_q / task_num # Average Loss
        meta_optim.zero_grad()
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num) # Accuracy

        return accs, overall_loss.detach().cpu().numpy(), network, meta_optim  


    def validation(self, x_spt, y_spt, x_qry, y_qry, model):
        
        querysz = x_qry.size(0)
        network = deepcopy(model) # copy the meta-parameters not to update during Validation/Test
        
        weights = OrderedDict((key, params) for (key, params) in network.named_parameters())

        for k in range(self.update_step_test): # Iterate the # of Inner-Loop steps (validation/test gradient steps -> 10)
            if k == 0 :
                logits = network(x_spt, weights)
                loss_ = self.loss(logits, y_spt)
                grad = torch.autograd.grad(loss_, weights.values(), create_graph=True)
                updated_weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(weights.items(), grad))
            else:
                logits = network(x_spt, updated_weights)
                loss_ = self.loss(logits, y_spt)
                grad = torch.autograd.grad(loss_, updated_weights.values(), create_graph=True)
                torch.nn.utils.clip_grad_norm_(updated_weights.values(), 5)
                updated_weights = OrderedDict((key, param - self.update_lr*grad) for ((key, param), grad) in zip(updated_weights.items(), grad)) 
            
        with torch.no_grad():
            logits_q = network(x_qry, updated_weights)
            loss_q = self.loss(logits_q, y_qry)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item() 
        
        accs = np.array(correct) / querysz # Accuracy

        return accs, loss_q.detach().cpu().numpy()
    
def main():
    pass

if __name__ == '__main__':
    main()
