import glob
import logging
import os
import random

import numpy
import soundfile
import torch
from scipy import signal
from torch.utils.data import Dataset

from voxceleb_util import read_train_list


def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    data1 = numpy.stack([audio], axis=0)

    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')

    if evalmode:
        start_frame = numpy.linspace(0, audio.shape[0] - max_audio, num=num_eval)
    else:
        start_frame = numpy.array([numpy.int64(random.random() * (audio.shape[0] - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in start_frame:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = numpy.stack(feats, axis=0)
    if evalmode:
        return data1, feat
    else:
        return feat


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, num_frames):
        self.num_frames = num_frames
        self.max_audio = max_audio = num_frames * 160 + 240
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
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
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
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float32), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class VoxCelebDataset(Dataset):
    def __init__(self, data_list, data_path, musan_path, rir_path, num_frames, keep_in_memory=False,
                 dtype=numpy.float32):
        super(VoxCelebDataset, self).__init__()

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, num_frames=num_frames)

        self.data_list = data_list
        self.data_path = data_path
        self.num_frames = num_frames
        self.keep_in_memory = keep_in_memory
        self.dtype = dtype

        speaker_id, self.wave_file_name = read_train_list(data_list)

        speaker_id_unique = numpy.unique(speaker_id)
        self.speaker_num = len(speaker_id_unique)
        label = {speaker: idx for idx, speaker in enumerate(speaker_id_unique)}
        speaker_label_list = [label[speaker] for speaker in speaker_id]
        self.speaker_label = numpy.array(speaker_label_list)

        if self.keep_in_memory:
            logging.info('Loading all data ...... ')
            self._load_all_()

    def _load_all_(self):
        tempfilename = os.path.join(self.data_path, self.wave_file_name[0])
        tempfeature = loadWAV(tempfilename, self.num_frames, evalmode=False)
        feat_size = tempfeature.shape

        self.feature = numpy.zeros((len(self.wave_file_name), feat_size[1]), dtype=self.dtype)
        for idx in range(len(self.wave_file_name)):
            data = self._loadone_(idx)
            self.feature[idx] = data
        self.feature = torch.from_numpy(self.feature)
        logging.info('VoxcelebDataset : shape = {}  type : {}'.format(self.feature.shape, self.feature.dtype))

    def _loadone_(self, index):
        wave_filename = os.path.join(self.data_path, self.wave_file_name[index])
        audio = loadWAV(wave_filename, self.num_frames, evalmode=False)

        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.augment_wav.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.augment_wav.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.augment_wav.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.augment_wav.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.augment_wav.add_noise(audio, 'speech')
            audio = self.augment_wav.add_noise(audio, 'music')

        return audio.squeeze(0)

    def __getitem__(self, index):
        if self.keep_in_memory:
            return self.feature[index], self.speaker_label[index]

        label = self.speaker_label[index]
        data = self._loadone_(index)

        return torch.FloatTensor(data), label

    def __len__(self):
        return len(self.wave_file_name)
