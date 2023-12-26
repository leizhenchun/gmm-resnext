########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################

import logging
import os
import shutil
import sys
import zipfile

import h5py
import numpy
import pandas as pd
import torch


def read_vox1_meta(meta_file):
    # VoxCeleb1 ID	VGGFace1 ID	Gender	Nationality	Set
    protocol = pd.read_csv(meta_file, header=0, sep='\t', dtype=str, engine='python',
                           names=['VoxCeleb1_ID', 'VGGFace1_ID', 'Gender', 'Nationality', 'Set'])
    VoxCeleb1_ID = protocol['VoxCeleb1_ID']
    VGGFace1_ID = protocol['VGGFace1_ID']
    gender = protocol['Gender']
    nationality = protocol['Nationality']
    dataset = protocol['Set']

    return VoxCeleb1_ID.values, VGGFace1_ID.values, gender.values, nationality.values, dataset.values


def read_vox2_meta(meta_file):
    # VoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set
    protocol = pd.read_csv(meta_file, header=0, sep=r' ,', dtype=str, engine='python',
                           names=['VoxCeleb2_ID', 'VGGFace2_ID', 'Gender', 'Set'])
    VoxCeleb2_ID = protocol['VoxCeleb2_ID']
    VGGFace2_ID = protocol['VGGFace2_ID']
    gender = protocol['Gender']
    dataset = protocol['Set']

    return VoxCeleb2_ID.values, VGGFace2_ID.values, gender.values, dataset.values


def read_train_list(train_list_filename):
    train_list = pd.read_csv(train_list_filename, header=None, sep=r' ', dtype=str, engine='python',
                             names=['speaker_id', 'wave_file_name'])
    speaker_id = train_list['speaker_id']
    wave_file_name = train_list['wave_file_name']
    return speaker_id.values, wave_file_name.values


def read_list_test(ist_test_filename):
    list_test = pd.read_csv(ist_test_filename, header=None, sep=r' ', dtype=str, engine='python',
                            names=['wave1', 'wave2'])
    wave1 = list_test['wave1']
    wave2 = list_test['wave2']
    return wave1.values, wave2.values


def read_veri_test(veri_test_filename):
    list_test = pd.read_csv(veri_test_filename, header=None, sep=r' ', dtype=str, engine='python',
                            names=['key', 'wave1', 'wave2'])
    key = list_test['key']
    wave1 = list_test['wave1']
    wave2 = list_test['wave2']
    return key.values, wave1.values, wave2.values


def save_scores(score_filename, wave1, wave2, scores):
    with open(score_filename, 'wt') as f:
        for i in range(len(wave1)):
            f.write('%s %s %.6f\n' % (wave1[i], wave2[i], scores[i]))


def read_iden_split(filename):
    protocol = pd.read_csv(filename, header=None, sep=r' ,', dtype=str, engine='python', names=['key', 'wavefile'])
    VoxCeleb2_ID = protocol['key']
    VGGFace2_ID = protocol['wavefile']

    return VoxCeleb2_ID.values, VGGFace2_ID.values


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def random_seed(seed=0):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        # os.mkdir(path_name)
        os.makedirs(dir_name)


def clear_dir(dir_name):
    logging.info('Clear dir : ' + dir_name)
    shutil.rmtree(dir_name, ignore_errors=True)
    os.makedirs(dir_name)


def load_array_h5(filename, data_name='data'):
    with h5py.File(filename, 'r') as f:
        data = f[data_name][:]
    return data


def save_array_h5(filename, data, data_name='data'):
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        f[data_name] = data


def zip_file(score_file, compresslevel=9):
    file_path, file_name = os.path.split(score_file)
    short_name, extension = os.path.splitext(file_name)
    # file_name, file_extension = os.path.splitext(score_file)

    try:
        with zipfile.ZipFile(file_path + '/' + short_name + '.zip', mode="w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=compresslevel) as f:
            f.write(score_file, arcname=short_name + '.txt')
    except Exception as e:
        logging.info(e)
    finally:
        f.close()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def get_parameter_number_details(net):
    trainable_num_details = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}
    return trainable_num_details


def show_model(model):
    logging.info(model)
    # model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logging.info('Model total parameter: {:,}'.format(model_params))

    logging.info(get_parameter_number_details(model))
    total_num, trainable_num = get_parameter_number(model)
    logging.info('Total params: {:,}      Trainable params: {:,}'.format(total_num, trainable_num))


if __name__ == '__main__':
    # train_file, train_type = read_protocol(r'Z:\asvspoof\DS_10283_3055\protocol_V2\ASVspoof2017_V2_train.trn.txt')
    # train_feat = load_feat_list(r'D:\LZC\asvspoof\feat\spoof2017_CQCC_train.h5', train_file)

    filename = '/home/lzc/lzc/ASVspoof2019/ASVspoof2019feat2/LFCC/LFCC_LA_train/LA_T_1000137.h5'
    zip_file(filename)
