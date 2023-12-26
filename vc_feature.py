import glob
import logging
import os
import random
import shutil

import h5py
import numpy
import soundfile
import torch
import torchaudio
from scipy import signal

from voxceleb_core.voxceleb_util import read_train_list
from voxceleb_process import PreEmphasis

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

musan_path = '/home/lzc/lzc/voxceleb/musan_split'
rir_path = '/home/lzc/lzc/voxceleb/RIRS_NOISES/simulated_rirs'


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path):
        # self.num_frames = num_frames
        # self.max_audio = max_audio = num_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def add_noise(self, audio, noisecat):
        # audio = numpy.stack([audio], axis=0)
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            # length = self.num_frames * 160 + 240
            length = audio.shape[-1]
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio

    def add_rev(self, audio):
        # audio = numpy.stack([audio], axis=0)
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        # return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]
        return signal.convolve(audio, rir, mode='full')[:, :audio.shape[-1]]


# class train_feature(object):
#     def __init__(self, train_list, train_path, musan_path, rir_path, augment=None, num_frames=200):
#         self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, num_frames=num_frames)
#         self.train_list = train_list
#         self.train_path = train_path
#         self.musan_path = musan_path
#         self.rir_path = rir_path
#         self.augment = augment
#         self.num_frames = num_frames
#
#         # Load data & labels
#         self.data_list = []
#         self.data_label = []
#         lines = open(self.train_list).read().splitlines()
#         dictkeys = list(set([x.split()[0] for x in lines]))
#         dictkeys.sort()
#         dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
#         for index, line in enumerate(lines):
#             speaker_label = dictkeys[line.split()[0]]
#             file_name = os.path.join(train_path, line.split()[1])
#             self.data_label.append(speaker_label)
#             self.data_list.append(file_name)
#         print(len(self.data_list))
#
#     def feat(self):
#         for index, file_name in enumerate(self.data_list):
#             # Read the utterance and randomly select the segment
#             audio, sr = soundfile.read(file_name)
#             length = self.num_frames * 160 + 240
#             if audio.shape[0] <= length:
#                 shortage = length - audio.shape[0]
#                 audio = numpy.pad(audio, (0, shortage), 'wrap')
#             start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
#             audio = audio[start_frame:start_frame + length]
#             audio = numpy.stack([audio], axis=0)
#             if self.augment:
#                 # Data Augmentation
#                 augtype = random.randint(0, 5)
#                 if augtype == 0:  # Original
#                     audio = audio
#                 elif augtype == 1:  # Reverberation
#                     audio = self.augment_wav.add_rev(audio)
#                 elif augtype == 2:  # Babble
#                     audio = self.augment_wav.add_noise(audio, 'speech')
#                 elif augtype == 3:  # Music
#                     audio = self.augment_wav.add_noise(audio, 'music')
#                 elif augtype == 4:  # Noise
#                     audio = self.augment_wav.add_noise(audio, 'noise')
#                 elif augtype == 5:  # Television noise
#                     audio = self.augment_wav.add_noise(audio, 'speech')
#                     audio = self.augment_wav.add_noise(audio, 'music')
#             # fbank = extract_fbank(audio)
#             # mfcc = extract_39mfcc(audio)
#             mfcc = extract_80mfcc(audio)
#             # lfcc = extract_lfcc(audio)
#             mfcc = mfcc - torch.mean(mfcc, dim=-1, keepdim=True)
#             # lfcc = lfcc - torch.mean(lfcc, dim=-1, keepdim=True)
#
#             with h5py.File(file_name[:-4] + '.h5', 'w') as f:
#                 f['/data'] = mfcc
#             print(index)


# class train_feature1(object):
#     def __init__(self, train_list, train_path, musan_path, rir_path, augment=None, num_frames=200):
#         self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, num_frames=num_frames)
#         self.train_list = train_list
#         self.train_path = train_path
#         self.musan_path = musan_path
#         self.rir_path = rir_path
#         self.augment = augment
#         self.num_frames = num_frames
#
#         # Load data & labels
#         self.data_list = []
#         self.data_label = []
#         lines = open(self.train_list).read().splitlines()
#         dictkeys = list(set([x.split()[0] for x in lines]))
#         dictkeys.sort()
#         dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
#         files = glob.glob('%s/*/*.flac' % self.train_path)
#         for index, file in enumerate(files):
#             id = file[59:66]
#             speaker_label = dictkeys[id]
#             self.data_label.append(speaker_label)
#             self.data_list.append(file)
#         print(len(self.data_list))
#
#     def feat(self):
#         for index, file_name in enumerate(self.data_list):
#             # Read the utterance and randomly select the segment
#             audio, sr = soundfile.read(file_name)
#             length = self.num_frames * 160 + 240
#             if audio.shape[0] <= length:
#                 shortage = length - audio.shape[0]
#                 audio = numpy.pad(audio, (0, shortage), 'wrap')
#             start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
#             audio = audio[start_frame:start_frame + length]
#             audio = numpy.stack([audio], axis=0)
#             if self.augment:
#                 # Data Augmentation
#                 augtype = random.randint(0, 5)
#                 if augtype == 0:  # Original
#                     audio = audio
#                 elif augtype == 1:  # Reverberation
#                     audio = self.augment_wav.add_rev(audio)
#                 elif augtype == 2:  # Babble
#                     audio = self.augment_wav.add_noise(audio, 'speech')
#                 elif augtype == 3:  # Music
#                     audio = self.augment_wav.add_noise(audio, 'music')
#                 elif augtype == 4:  # Noise
#                     audio = self.augment_wav.add_noise(audio, 'noise')
#                 elif augtype == 5:  # Television noise
#                     audio = self.augment_wav.add_noise(audio, 'speech')
#                     audio = self.augment_wav.add_noise(audio, 'music')
#             # fbank = extract_fbank(audio)
#             # mfcc = extract_39mfcc(audio)
#             mfcc = extract_80mfcc(audio)
#             # lfcc = extract_lfcc(audio)
#             # mfcc = mfcc - torch.mean(mfcc, dim=-1, keepdim=True)
#             mfcc = mfcc - torch.mean(mfcc, dim=-1, keepdim=True)
#
#             with h5py.File(file_name[:-5] + '.h5', 'w') as f:
#                 f['/data'] = mfcc
#             print(index)


# class test_feature(object):
#     def __init__(self, test_list, test_path, num_frames=300):
#         self.test_list = test_list
#         self.test_path = test_path
#         self.num_frames = num_frames
#         self.files = []
#         lines = open(self.test_list).read().splitlines()
#         for line in lines:
#             self.files.append(line.split()[1])
#             self.files.append(line.split()[2])
#         self.setfiles = list(set(self.files))
#         self.setfiles.sort()
#         print(len(self.setfiles))
#
#     def feat(self):
#         for index, file_name in enumerate(self.setfiles):
#             file_name = os.path.join(self.test_path, file_name)
#             audio, sr = soundfile.read(file_name)
#             # max_audio = self.num_frames * 160 + 240
#             # if audio.shape[0] <= max_audio:
#             #     shortage = max_audio - audio.shape[0]
#             #     audio = numpy.pad(audio, (0, shortage), 'wrap')
#             # feats = []
#             # startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
#             # for asf in startframe:
#             #     feats.append(audio[int(asf):int(asf) + max_audio])
#             # fbank = extract_fbank(audio).squeeze(0)
#             mfcc = extract_39mfcc(audio)
#             with h5py.File(file_name[:-4] + '.h5', 'w') as f:
#                 f['/data'] = mfcc
#             print(index)


def extract_fbank(audio):
    torchfbank = torch.nn.Sequential(
        # PreEmphasis(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                             f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80))
    audio = torch.FloatTensor(audio)
    fbank = torchfbank(audio).squeeze(0)

    return fbank


def extract_39mfcc(audio):
    torchmfcc = torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13,
                                   melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                                              "window_fn": torch.hamming_window, "n_mels": 13}))
    audio = torch.FloatTensor(audio)
    mfcc = torchmfcc(audio).squeeze(0)
    delta1 = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta1)
    feature = numpy.concatenate((mfcc, delta1, delta2), axis=0)

    return feature


def extract_80mfcc(audio):
    torchmfcc = torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=80,
                                   melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                                              "window_fn": torch.hamming_window, "n_mels": 80}))
    audio = torch.FloatTensor(audio)
    mfcc = torchmfcc(audio).squeeze(0)
    # mfcc = torchmfcc(audio)
    return mfcc


def extract_lfcc(audio):
    torchlfcc = torch.nn.Sequential(
        PreEmphasis(),
        torchaudio.transforms.LFCC(sample_rate=16000, n_filter=80, n_lfcc=80,
                                   speckwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
                                               "window_fn": torch.hamming_window}))
    audio = torch.FloatTensor(audio)
    lfcc = torchlfcc(audio).squeeze(0)

    return lfcc


def extract_train_list(train_list_file, train_wave_path, train_feat_path):
    logging.info('Clear dir : ' + train_feat_path)
    shutil.rmtree(train_feat_path, ignore_errors=True)
    os.makedirs(train_feat_path)

    speaker_id, wave_file_name = read_train_list(train_list_file)

    extract_list(wave_file_name, train_wave_path, train_feat_path)


def extract_list(wave_list, wave_path, feat_path):
    augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path)

    for idx in range(len(wave_list)):
        wave_file = wave_list[idx]

        audio, sr = soundfile.read(os.path.join(wave_path, wave_file))
        audio = audio[numpy.newaxis, :]
        # length = self.num_frames * 160 + 240
        # if audio.shape[0] <= length:
        #     shortage = length - audio.shape[0]
        #     audio = numpy.pad(audio, (0, shortage), 'wrap')
        # start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        # audio = audio[start_frame:start_frame + length]
        # audio = numpy.stack([audio], axis=0)
        # if True:
        # Data Augmentation

        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = augment_wav.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = augment_wav.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = augment_wav.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = augment_wav.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = augment_wav.add_noise(audio, 'speech')
            audio = augment_wav.add_noise(audio, 'music')
        # fbank = extract_fbank(audio)
        # mfcc = extract_39mfcc(audio)
        mfcc = extract_80mfcc(audio)
        # lfcc = extract_lfcc(audio)
        # mfcc = mfcc - torch.mean(mfcc, dim=-1, keepdim=True)
        mfcc = mfcc - torch.mean(mfcc, dim=-1, keepdim=True)

        feat_file_name = os.path.join(feat_path, wave_file[:-3] + 'h5')
        feat_dir = os.path.dirname(feat_file_name)
        os.makedirs(feat_dir, exist_ok=True)
        with h5py.File(feat_file_name, 'w') as f:
            f['/data'] = mfcc.numpy().astype(numpy.float32)

        if idx % 10000 == 0:
            logging.info(r'{:,}/{:,}'.format(idx, len(wave_list)))

    logging.info('Done!')


if __name__ == '__main__':
    train_list_file = r'/home/lzc/lzc/voxceleb/train_list.txt'
    train_wave_path = r'/home/lzc/lzc/voxceleb/voxceleb2/'
    train_feat_path = r'/home/lzc/lzc/voxceleb/voxceleb2_mfcc80/'
    extract_train_list(train_list_file, train_wave_path, train_feat_path)
