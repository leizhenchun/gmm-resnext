########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import sys
import time
import warnings
from collections import OrderedDict
from shutil import copy

import numpy
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from voxceleb_dataset import VoxCelebDataset, loadWAV
from voxceleb_eval_metrics import ComputeErrorRates, tuneThresholdfromScore, ComputeMinDcf
from voxceleb_logging import logger_init_basic, MyLogger
from voxceleb_loss import AAMsoftmax
from voxceleb_util import make_dir, is_debug, show_model
from voxceleb_util import read_veri_test, save_scores


def get_parameter():
    params = {}

    params['voxceleb_data_path'] = '/home/lzc/lzc/voxceleb/'
    params['voxceleb_exp_path'] = '/home/lzc/lzc/voxceleb_exp/'

    params['batch_size'] = 20

    params['feature_num'] = 200
    params['feature_num_test'] = 400
    params['m'] = 0.2
    params['s'] = 30
    params['n_class'] = 5994

    # params['gmm_type'] = 'ms_gmm'
    params['groups'] = 1

    params['lr'] = 0.001
    params['weight_decay'] = 2e-5

    params['test_epoch_from'] = 50
    params['test_epoch_step'] = 5

    params['use_scheduler'] = True
    params['scheduler_step_size'] = 1
    params['scheduler_gamma'] = 0.97

    params['num_epochs'] = 1

    params['train_model'] = True

    params['test_VoxCeleb_O'] = True
    params['test_VoxCeleb_E'] = True
    params['test_VoxCeleb_H'] = True

    params['save_model'] = False
    params['save_score'] = True
    params['verbose'] = 1

    return params


def parameter_2_str(parameter):
    p = OrderedDict(parameter)
    result = 'nn_parameter:\n'
    for key in p:
        result += '{} : {}\n'.format(key, p[key])
    return result


class VoxCelebExperiment:
    def __init__(self, model_type, feature_type, parm=get_parameter()):

        self.model_type = model_type
        self.feature_type = feature_type
        self.verbose = parm['verbose']
        self.parm = parm

        self.num_epochs = parm['num_epochs']
        self.use_scheduler = parm['use_scheduler']
        self.test_epoch_from = parm['test_epoch_from']
        self.test_epoch_step = parm['test_epoch_step']

        self.use_gpu = torch.cuda.is_available()
        self.is_debug = is_debug()

        self.exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        self.dataset_cls = VoxCelebDataset

        self._path_init()

        make_dir(self.exp_path)
        copy(sys.argv[0], self.exp_path)

        if self.is_debug:
            logger_init_basic()
        else:
            sys.stdout = MyLogger(self.exp_path, model_type, feature_type)
        logging.info('\n\n======== {} {} ========'.format(self.model_type, self.feature_type))
        logging.info(parameter_2_str(self.parm))

        self.bestEER = 100.0
        self.bestMinDCF = 100.0

    def _path_init(self):

        model_feat = self.model_type + '_' + self.feature_type

        self.data_path = self.parm['voxceleb_data_path']
        self.exp_path = os.path.join(self.parm['voxceleb_exp_path'], model_feat, self.exp_time + '_' + model_feat)

        self.model_file = self.exp_path + '/VC_' + model_feat + '_model.pkl'
        self.score_file = self.exp_path + '/VC_' + model_feat + '_{}_score.txt'
        self.output_file = self.exp_path + '/VC_' + model_feat + '_{}_output.h5'

        self.train_list = os.path.join(self.data_path, 'train_list.txt')
        self.veri_test2 = os.path.join(self.data_path, 'veri_test2.txt')
        self.list_test_all2 = os.path.join(self.data_path, 'list_test_all2.txt')
        self.list_test_hard2 = os.path.join(self.data_path, 'list_test_hard2.txt')

        self.musan_path = os.path.join(self.data_path, 'musan_split')
        self.rir_path = os.path.join(self.data_path, 'RIRS_NOISES/simulated_rirs')

        self.train_wave_path = os.path.join(self.data_path, 'voxceleb2')
        # self.test_wave_path = os.path.join(self.data_path, 'vox1_test_wav/wav')

        return

    def run(self):
        warnings.simplefilter("ignore")
        if self.parm['train_model']:
            logging.info(
                '========== VoxCeleb Train Model: {} {} ......'.format(self.model_type, self.feature_type))
            self.model = self.train()
            logging.info("========== Done!")

        if self.parm['test_VoxCeleb_O']:
            logging.info(
                '========== VoxCeleb Test on VoxCeleb_O: {} {} ......'.format(self.model_type, self.feature_type))
            self.test(self.model, self.veri_test2, 'VoxCeleb_O')
            logging.info("========== Done!")

        if self.parm['test_VoxCeleb_E']:
            logging.info(
                '========== VoxCeleb Test on VoxCeleb_E: {} {} ......'.format(self.model_type, self.feature_type))
            self.test(self.model, self.list_test_all2, 'VoxCeleb_E')
            logging.info("========== Done!")

        if self.parm['test_VoxCeleb_H']:
            logging.info(
                '========== VoxCeleb Test on VoxCeleb_H: {} {} ......'.format(self.model_type, self.feature_type))
            self.test(self.model, self.list_test_hard2, 'VoxCeleb_H')
            logging.info("========== Done!")

    def train(self):

        logging.info("Modeling .......")
        model = self.get_net()
        if self.use_gpu:
            model = model.cuda()
        model.train()
        show_model(model)

        if self.num_epochs <= 0:
            return model

        train_dataset = self.dataset_cls(data_list=self.train_list,
                                         data_path=self.train_wave_path,
                                         musan_path=self.musan_path,
                                         rir_path=self.rir_path,
                                         num_frames=self.parm['feature_num'])
        train_loader = DataLoader(train_dataset, batch_size=self.parm['batch_size'], shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
        logging.info('Dataset Length : {}   DataLoader Length : {}'.format(len(train_dataset), len(train_loader)))

        self.train_loader_len = len(train_loader)

        self.train_model(model, train_loader)

        if 'save_model' in self.parm and self.parm['save_model']:
            make_dir(self.exp_path)
            logging.info("Saving Model to: " + self.model_file)
            torch.save(model, self.model_file)

        return model

    def train_model(self, model, train_loader):
        self.optimizer = optimizer = self.get_optimizer(model)
        self.scheduler = scheduler = self.get_scheduler(optimizer)
        self.criterion = criterion = self.get_criterion()
        if self.use_gpu:
            criterion = criterion.cuda()
        logging.info('Optimizer : ' + str(optimizer))
        logging.info('Scheduler : ' + str(scheduler))
        logging.info('Criterion : ' + str(criterion))

        model.train()

        logging.info("Training ......")
        for epoch in range(1, self.num_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(model, optimizer, criterion, scheduler, train_loader, epoch)

            if epoch >= self.test_epoch_from and epoch % self.test_epoch_step == 0:
                self.test(model, self.veri_test2, 'VoxCeleb_O_EPOCH')

    def train_one_epoch(self, model, optimizer, criterion, scheduler, train_loader, epoch):
        model.train()

        clr = [param['lr'] for param in optimizer.param_groups]
        total = 0
        total_loss = 0.0
        total_correct = 0.0
        for num, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            # output = model(data.cuda())

            # nloss, prec = criterion(output, labels)

            nloss, prec = model(data.cuda(), labels)

            nloss.backward()
            optimizer.step()

            total += labels.size(0)
            total_correct += prec.item() * labels.size(0)
            total_loss += nloss.detach().cpu().numpy()

        if self.parm['use_scheduler']:
            self.scheduler_step(scheduler, total_loss)

        if self.verbose >= 1:
            logging.info(
                "Train Epoch : {}/{}  LR = {:.8f}  ACC = {:.4f}%  LOSS = {:.6f}".format(epoch, self.num_epochs,
                                                                                        max(clr),
                                                                                        total_correct / total,
                                                                                        total_loss / total,
                                                                                        ))

    def test(self, model, test_list, test_type):
        model.eval()

        # if test_type in ('VoxCeleb_O', 'VoxCeleb_O_EPOCH'):
        #     test_wave_path = os.path.join(self.data_path, 'vox1_test_wav/wav')
        # else:
        #     test_wave_path = os.path.join(self.data_path, 'vox1_dev_wav/wav')

        test_wave_path = os.path.join(self.data_path, 'voxceleb1')

        key_list, wave1_list, wave2_list = read_veri_test(test_list)

        key_list = [int(i) for i in key_list]
        wave_all = numpy.concatenate((wave1_list, wave2_list))
        wave_all = numpy.unique(wave_all)

        if not test_type == 'VoxCeleb_O_EPOCH':
            logging.info('Testing ......    {:,} / {:,}'.format(len(wave_all), len(key_list)))

        feature_num_test = self.parm['feature_num_test']

        embeddings = {}
        for wave in wave_all:
            filename = os.path.join(test_wave_path, wave)
            audio1, audio2 = loadWAV(filename, feature_num_test, evalmode=True)
            audio1 = torch.FloatTensor(audio1).cuda()
            audio2 = torch.FloatTensor(audio2).cuda()
            with torch.no_grad():
                embedding_1 = model(audio1)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = model(audio2)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[wave] = [embedding_1, embedding_2]

        scores = []
        for index in range(len(key_list)):
            wave1 = wave1_list[index]
            wave2 = wave2_list[index]

            embedding_11, embedding_12 = embeddings[wave1]
            embedding_21, embedding_22 = embeddings[wave2]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2

            score = score.detach().cpu().numpy()
            scores.append(score)

        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, key_list, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, key_list)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

        if test_type == 'VoxCeleb_O_EPOCH':
            if self.bestEER > EER: self.bestEER = EER
            if self.bestMinDCF > minDCF: self.bestMinDCF = minDCF
            logging.info(
                '    EER : {:.4f}%,     minDCF : {:.6f}  [Best EER : {:.4f}%,  Best minDCF : {:.6f}]'.format(EER,
                                                                                                             minDCF,
                                                                                                             self.bestEER,
                                                                                                             self.bestMinDCF))
            return
        else:
            logging.info('      EER: {:.4f}%,       minDCF: {:.6f}'.format(EER, minDCF))

        if 'save_score' in self.parm and self.parm['save_score']:
            test_score_file = self.score_file.format(test_type)
            logging.info('Saving Scores to:' + test_score_file)
            save_scores(test_score_file, wave1_list, wave2_list, scores)

    def get_net(self, num_classes=5994):
        return None

    def get_criterion(self):
        return AAMsoftmax(n_class=self.parm['n_class'], m=self.parm['m'], s=self.parm['s'])

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.parm['lr'], weight_decay=self.parm['weight_decay'])

    def get_scheduler(self, optimizer):
        return StepLR(optimizer, step_size=self.parm['scheduler_step_size'], gamma=self.parm['scheduler_gamma'])

    # def scheduler_step(self, scheduler, epoch):
    #     scheduler.step(epoch - 1)

    def scheduler_step(self, scheduler, loss):
        scheduler.step()
        # scheduler.step(loss)


if __name__ == '__main__':
    exp_param = get_parameter()

    model_type = 'GMM_ResNet'
    feature_type = 'MFCC'

    VoxCelebExperiment(model_type, feature_type, parm=exp_param).run()
