'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from voxceleb_eval_metrics import accuracy


# class Classifier(nn.Module):
#     def __init__(self, input_dim, out_neurons=1000):
#         super().__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(out_neurons, input_dim))
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, x):
#         x = F.linear(F.normalize(x), F.normalize(self.weight))
#         return x


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 256), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        # nn.init.xavier_normal_(self.weight, gain=1.0)
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1


class AMsoftmax(nn.Module):
    def __init__(self, n_class, m, s, **kwargs):
        super(AMsoftmax, self).__init__()

        self.m = m
        self.s = s
        self.in_feats = 192
        self.W = torch.nn.Parameter(torch.randn(192, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        # print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        prec1 = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1


class softmax(nn.Module):
    def __init__(self, n_class, **kwargs):
        super(softmax, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(512, n_class)

        # print('Initialised Softmax Loss')

    def forward(self, x, label=None):
        x = self.fc(x)
        nloss = self.criterion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
