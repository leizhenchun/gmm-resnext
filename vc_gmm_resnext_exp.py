########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################
import logging

import torch
import torch.nn as nn
import torchaudio
import torchinfo
import numpy
from torch.utils.data import DataLoader, TensorDataset
import torchsummary
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from voxceleb_core.voxceleb_experiment import VoxCelebExperiment, get_parameter
from voxceleb_core.voxceleb_process import PreEmphasis
from voxceleb_core.voxceleb_gmm import GMM, GMMLayer
from voxceleb_core.voxceleb_loss import AAMsoftmax
from voxceleb_core.voxceleb_util import make_dir, is_debug, show_model


def exp_parameters():
    exp_param = get_parameter()

    exp_param['voxceleb_data_path'] = '/home/labuser/ssd/yanhui/data/min_voxceleb/'
    exp_param['voxceleb_exp_path'] = '/home/labuser/yanhui/voxceleb_exp/'

    exp_param['batch_size'] = 2

    # exp_param['feature_size'] = 80
    exp_param['feature_num'] = 200
    exp_param['feature_num_test'] = 400
    exp_param['m'] = 0.2
    exp_param['s'] = 30

    exp_param['n_class'] = 5994

    # exp_param['gmm_size'] = 512
    # exp_param['groups'] = 1

    exp_param['weight_decay'] = 2e-5

    exp_param['use_scheduler'] = True
    exp_param['scheduler_step_size'] = 1
    exp_param['scheduler_gamma'] = 0.97

    exp_param['test_epoch_from'] = 0
    exp_param['test_epoch_step'] = 1

    exp_param['num_epochs'] = 100

    exp_param['lr'] = 0.001

    exp_param['test_VoxCeleb_O'] = True
    exp_param['test_VoxCeleb_E'] = True
    exp_param['test_VoxCeleb_H'] = True

    exp_param['save_model'] = False
    exp_param['save_score'] = True
    exp_param['verbose'] = 1

    return exp_param


class embedding_fusion(nn.Module):
    def __init__(self, inplanes, planes, num_class=5994):
        super(embedding_fusion, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.aam = AAMsoftmax(n_class=5994, m=0.2, s=30)

    def forward(self, x, label=None):
        x = self.fc(x)
        if self.training:
            return self.aam(x, label)

        return x


class TwoPathModel(nn.Module):
    def __init__(self, path1, path2, embedding_layer):
        super(TwoPathModel, self).__init__()
        self.path1 = path1
        self.path2 = path2
        self.embedding_layer = embedding_layer

    def forward(self, x, label=None):
        x1 = self.path1(x, label)
        x2 = self.path2(x, label)

        output = torch.cat((x1, x2), dim=1)
        output = self.embedding_layer(output, label)

        return output


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128, groups=1):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0, groups=groups),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0, groups=groups),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class SEBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4, groups=1, group_width=512):
        super(SEBottleneck, self).__init__()
        mid_planes = groups * group_width

        self.conv1 = nn.Conv1d(inplanes, mid_planes, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_planes)
        self.conv2 = nn.Conv1d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(mid_planes)
        self.conv3 = nn.Conv1d(mid_planes, planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.se = SEModule(planes, bottleneck=32)
        # self.se = SEModule(planes, bottleneck=64)
        self.se = SEModule(planes, bottleneck=128)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GMMResNext(nn.Module):

    def __init__(self,
                 gmm,
                 planes,
                 layers,
                 num_class=5994,
                 groups=1,
                 group_width=512):
        super(GMMResNext, self).__init__()

        # self.layers = [3, 3, 9, 3]
        # self.sum_planes = [512, 512, 512, 512]
        # self.sum_planes = [256, 256, 256, 256]
        # self.sum_planes = [128, 128, 128, 128]

        self.extract_80mfcc = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80,
                                       melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                                                  "window_fn": torch.hamming_window, "n_mels": 80}))
        self.inplanes = len(gmm.w)
        self.gmm_layer = GMMLayer(gmm, requires_grad=True)

        self.conv1 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, groups=groups)
        # self.conv1 = nn.Conv1d(80, 512, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(self.inplanes)

        self.layer1 = self._make_layer(SEBottleneck, planes[0], layers[0], groups=groups, group_width=group_width)
        self.layer2 = self._make_layer(SEBottleneck, planes[1], layers[1], groups=groups, group_width=group_width)
        self.layer3 = self._make_layer(SEBottleneck, planes[2], layers[2], groups=groups, group_width=group_width)
        self.layer4 = self._make_layer(SEBottleneck, planes[3], layers[3], groups=groups, group_width=group_width)

        self.attention = nn.Sequential(
            nn.Conv1d(self.inplanes * 4 * 3, 128, kernel_size=1, groups=groups),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(128, self.inplanes * 4, kernel_size=1, groups=groups),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(self.inplanes * 4)
        self.fc6 = nn.Linear(self.inplanes * 4 * 2, 256)
        self.output_size = planes[-1]

        # self.classifier = Classifier(self.output_size, num_class)

        # self.aam = AAMsoftmax(n_class=self.parm['n_class'], m=self.parm['m'], s=self.parm['s'])
        self.aam = AAMsoftmax(n_class=5994, m=0.2, s=30)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, group_width=512):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, groups=groups, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=groups, group_width=group_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, group_width=group_width))

        return nn.Sequential(*layers)

    def forward(self, x, label=None):

        with torch.no_grad():
            x = self.extract_80mfcc(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.gmm_layer(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn5(x)

        t = x.size()[-1]

        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)

        x = self.fc6(x)

        if self.training:
            return self.aam(x, label)

        return x


class GMMResNextExperiment(VoxCelebExperiment):
    def __init__(self, model_type, feature_type, parm):
        super(GMMResNextExperiment, self).__init__(model_type, feature_type, parm=parm)

        self.gmm_ubm = GMM.load_gmm(self.parm['gmm_file'])

    def get_net(self, num_classes=5994):
        if self.model_type == 'GMM_ResNext':
            model = GMMResNext(self.gmm_ubm,
                               self.parm['planes'],
                               self.parm['layers'],
                               self.parm['n_class'],
                               groups=1,
                               group_width=512)
        model = model.cuda()

        torchinfo.summary(model, (2, 200 * 160 + 240), depth=5)
        # torchsummary.summary(model, (200 * 160 + 240,))

        return model

    def get_scheduler(self, optimizer):
        # return ReduceLROnPlateau(optimizer, patience=5, factor=0.1, min_lr=self.parm['min_lr'], threshold=1e-6,
        #                          verbose=True)

        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # [10, 30, 70, 150, 310, 630]

    # def scheduler_step(self, scheduler, loss):
    #     scheduler.step(loss)


class GMMResNext2P2SExperiment(VoxCelebExperiment):
    def __init__(self, model_type, feature_type, parm):
        super(GMMResNext2P2SExperiment, self).__init__(model_type, feature_type, parm=parm)

        self.gmm_male, self.gmm_female = GMM.load_gmm2p(self.parm['gmm_file'])

    def get_net(self, num_classes=5994):
        if self.model_type == 'GMM_ResNext2P2S':
            path1 = GMMResNext(self.gmm_male,
                                     self.parm['planes'],
                                     self.parm['layers'],
                                     self.parm['n_class'])
            path2 = GMMResNext(self.gmm_female,
                                     self.parm['planes'],
                                     self.parm['layers'],
                                     self.parm['n_class'])
            embedding_layer = embedding_fusion(path1.output_size * 2, path1.output_size, self.parm['n_class'])

        model = TwoPathModel(path1, path2, embedding_layer)
        model = model.cuda()

        torchinfo.summary(model, (2, 200 * 160 + 240), depth=5)

        return model

    def train_model(self, model, train_loader):
        self.train_resnext_blocks(model, train_loader)
        self.train_embedding_layer(model, train_loader)

    def train_resnext_blocks(self, model, train_loader):
        logging.info('=======Training ResNext 2 Paths ......')

        path_classifier1 = model.path1
        path_classifier2 = model.path2
        if self.use_gpu:
            path_classifier1 = path_classifier1.cuda()
            path_classifier2 = path_classifier2.cuda()
        show_model(path_classifier1)

        optimizer1 = self.get_optimizer(path_classifier1)
        optimizer2 = self.get_optimizer(path_classifier2)

        scheduler1 = self.get_scheduler(optimizer1)
        scheduler2 = self.get_scheduler(optimizer2)

        path_classifier1.train()
        path_classifier2.train()

        for epoch in range(1, self.num_epochs + 1):

            clr = [param['lr'] for param in optimizer1.param_groups]
            total1 = 0
            total_loss1 = 0.0
            total_correct1 = 0.0

            total2 = 0
            total_loss2 = 0.0
            total_correct2 = 0.0

            for num, (data, labels, idx) in enumerate(train_loader):
                # train path1
                optimizer1.zero_grad()
                labels = torch.LongTensor(labels).cuda()

                nloss1, prec1 = path_classifier1(data.cuda(), labels)

                nloss1.backward()
                optimizer1.step()

                total1 += labels.size(0)
                total_correct1 += prec1.item() * labels.size(0)
                total_loss1 += nloss1.detach().cpu().numpy()

                # train path2
                optimizer2.zero_grad()

                nloss2, prec2 = path_classifier2(data.cuda(), labels)

                nloss2.backward()
                optimizer2.step()

                total2 += labels.size(0)
                total_correct2 += prec2.item() * labels.size(0)
                total_loss2 += nloss2.detach().cpu().numpy()

            if self.parm['use_scheduler']:
                self.scheduler_step(scheduler1, total_loss1)
                self.scheduler_step(scheduler2, total_loss2)

            if self.verbose >= 1:
                logging.info(
                    "Train Epoch : {}/{}  LR = {:.8f}  ACC1 = {:.4f}%  LOSS1 = {:.6f}".format(epoch, self.num_epochs,
                                                                                              max(clr),
                                                                                              total_correct1 / total1,
                                                                                              total_loss1 / total1,
                                                                                              ))
                logging.info(
                    "Train Epoch : {}/{}  LR = {:.8f}  ACC2 = {:.4f}%  LOSS2 = {:.6f}".format(epoch, self.num_epochs,
                                                                                              max(clr),
                                                                                              total_correct2 / total2,
                                                                                              total_loss2 / total2,
                                                                                              ))

    def train_embedding_layer(self, model, train_loader):
        logging.info('=======Training embedding_layer ......')

        model.path1.eval()
        model.path2.eval()

        output = None
        label_id = torch.zeros((len(train_loader.dataset),), dtype=torch.int64)

        with torch.no_grad():
            for num, (data, labels, idx) in enumerate(train_loader, start=1):
                labels = torch.LongTensor(labels).cuda()

                data1 = model.path1(data.cuda())
                data2 = model.path2(data.cuda())
                data = torch.cat((data1, data2), dim=1)

                if output is None:
                   output = torch.zeros((len(train_loader.dataset), data.shape[1]))
                output[idx, :] = data.cpu()
                label_id[idx] = labels.cpu()

        out_dataset = TensorDataset(output, label_id)
        loader = DataLoader(out_dataset, batch_size=200, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        logging.info(model.embedding_layer)
        optimizer = self.get_optimizer(model.embedding_layer)
        scheduler = self.get_scheduler(optimizer)

        model.embedding_layer.train()
        for epoch in range(1, self.num_epochs + 1):
            clr = [param['lr'] for param in optimizer.param_groups]
            total = 0
            total_loss = 0.0
            total_correct = 0.0

            for num, (data, labels) in enumerate(loader):
                optimizer.zero_grad()
                labels = torch.LongTensor(labels).cuda()

                nloss, prec = model.embedding_layer(data.cuda(), labels)

                nloss.backward()
                optimizer.step()

                total += labels.size(0)
                total_correct += prec.item() * labels.size(0)
                total_loss += nloss.detach().cpu().numpy()

            if self.parm['use_scheduler']:
                self.scheduler_step(scheduler, total_loss)

            if self.verbose >= 1:
                logging.info(
                    "Train Epoch : {}/{}  LR = {:.8f}  ACC1 = {:.4f}%  LOSS1 = {:.6f}".format(epoch, self.num_epochs,
                                                                                              max(clr),
                                                                                              total_correct / total,
                                                                                              total_loss / total,
                                                                                              ))

    def get_scheduler(self, optimizer):
        # return ReduceLROnPlateau(optimizer, patience=5, factor=0.1, min_lr=self.parm['min_lr'], threshold=1e-6,
        #                          verbose=True)

        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)  # [10, 30, 70, 150, 310, 630]


if __name__ == '__main__':
    exp_param = exp_parameters()
    exp_param['lr'] = 0.001
    exp_param['num_epochs'] = 1

    exp_param['planes'] = [512, 512, 512, 512, 256]
    exp_param['layers'] = [3, 3, 9, 3]
    exp_param['n_class'] = 5994
    model_type = 'GMM_ResNext'
    feature_type = 'MFCC'

    exp_param['gmm_file'] = '/home/labuser/ssd/yanhui/data/GMMexp/ms_gmm_mean_augment_80_mfcc/gmm_80_mfcc_512_llk_mean_std.h5'
    # exp_param['gmm_file'] = '/home/labuser/ssd/yanhui/data/GMMexp/ms_gmm_mean_augment2_80_mfcc/gmm_80_mfcc_512_llk_mean_std.h5'

    for _ in range(3):
        GMMResNextExperiment(model_type, feature_type, parm=exp_param).run()

    # model_type = 'GMM_ResNext2P2S'
    # for _ in range(3):
    #     GMMResNext2P2SExperiment(model_type, feature_type, parm=exp_param).run()