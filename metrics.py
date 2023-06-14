import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self, n_class, eps=1e-7):
        super(Accuracy, self).__init__()

        self.n_class = n_class
        self.eps = eps

    def forward(self, class_pred, class_gt):
        # flatten
        class_pred = class_pred.view(-1)
        class_gt = class_gt.view(-1)

        acc = 0

        for i in range(self.n_class): acc += torch.sum((class_pred[:] == i) * (class_gt[:] == i))

        acc = acc.to(torch.float)
        acc /= class_gt.shape[0]
        
        return acc


class MIoU(nn.Module):
    def __init__(self, n_class, eps=1e-7):
        super(MIoU, self).__init__()

        self.n_class = n_class
        self.eps = eps

    def forward(self, class_pred, class_gt):
        # flatten
        class_pred = class_pred.view(-1)
        class_gt = class_gt.view(-1)

        miou = 0

        for i in range(self.n_class):
            intersection = torch.sum((class_pred[:] == i) * (class_gt[:] == i))
            total = torch.sum(class_pred[:] == i) + torch.sum(class_gt[:] == i)
            union = total - intersection

            miou += (intersection + self.eps) / (union + self.eps)
            
        miou /= self.n_class
        
        return miou
    

class FwIoU(nn.Module):
    def __init__(self, n_class, eps=1e-7):
        super(FwIoU, self).__init__()

        self.n_class = n_class
        self.eps = eps

    def forward(self, class_pred, class_gt):
        # flatten
        class_pred = class_pred.view(-1)
        class_gt = class_gt.view(-1)

        fwiou = 0

        for i in range(self.n_class):
            intersection = torch.sum((class_pred[:] == i) * (class_gt[:] == i))
            total = torch.sum(class_pred[:] == i) + torch.sum(class_gt[:] == i)
            union = total - intersection

            fwiou += ( total / (2 * class_gt.shape[0]) ) * ( (intersection + self.eps) / (union + self.eps) )
        
        return fwiou