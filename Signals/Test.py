import argparse
import os
import random
import warnings
from glob import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, TensorDataset
import math
import scipy.io.wavfile as wf
from time import localtime, strftime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from Train import MyNet
from pprint import pprint

# INFO: Set random seeds
np.random.seed(4)
th.manual_seed(4)
th.cuda.manual_seed_all(4)
random.seed(4)


def strided_app(a, L, S):
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n))


def custom_loader(speechfolder,
                  eggfolder,
                  window,
                  stride,
                  subwindow=40,
                  select=None):
    speechfiles = sorted(glob(os.path.join(speechfolder, '*.npy')))
    eggfiles = sorted(glob(os.path.join(eggfolder, '*.npy')))

    if select is not None:
        ind = np.random.permutation(len(speechfiles))
        ind = ind[:select]
        speechfiles = [speechfiles[i] for i in ind]
        eggfiles = [eggfiles[i] for i in ind]
        print("Selected {} files".format(select))

    speech_data = [np.load(f) for f in speechfiles]
    egg_data = [np.load(f) for f in eggfiles]

    for i in range(len(egg_data)):
        egg_data[i] = egg_data[i] / np.max(np.abs(egg_data[i]))

    for i in range(len(speech_data)):
        speech_data[i] = speech_data[i] / np.max(np.abs(speech_data[i]))

    speech_data = np.concatenate(speech_data)
    egg_data = np.concatenate(egg_data)

    speech_windowed_data = strided_app(speech_data, window, stride)
    egg_windowed_data = strided_app(egg_data, window, stride)

    return speech_windowed_data, egg_windowed_data


def create_dataloader(batch_size,
                      speechfolder,
                      eggfolder,
                      window,
                      stride,
                      subwindow=32,
                      select=None):
    print(select, "files to be selected")
    speech_windowed_data, egg_windowed_data = custom_loader(
        speechfolder,
        eggfolder,
        window,
        stride,
        subwindow=subwindow,
        select=select)
    dataset = TensorDataset(
        th.from_numpy(np.expand_dims(speech_windowed_data, 1).astype(np.float32)),
        th.from_numpy(egg_windowed_data.astype(np.float32)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False)
    return dataloader


def test(model: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool = False,
         threshold: float = 0.5):
    if use_cuda:
        model.cuda()
    model.eval()

    dump_array_reconstructions = []
    dump_array_egg = []
    for data, egg_data in test_loader:
        if use_cuda:
            data, egg_data = data.cuda(), egg_data.cuda()
        data, egg_data = Variable(data, volatile=True), Variable(egg_data)

        reconstructions = model(data)
        regress = reconstructions[:, 1]

        x = np.zeros(egg_data.shape)
        x[:, regress.cpu().detach().numpy().astype(int)] = 1

        dump_array_reconstructions.append(
            x.ravel())
        dump_array_egg.append(egg_data.cpu().detach().numpy().ravel())

        del data, reconstructions
        th.cuda.empty_cache()

    dump_array_reconstructions = np.concatenate(dump_array_reconstructions)
    dump_array_egg = np.concatenate(dump_array_egg)

    plt.figure()
    plt.plot(np.arange(
        dump_array_reconstructions.shape[0]), dump_array_reconstructions, "b")
    plt.plot(np.arange(dump_array_egg.shape[0]), dump_array_egg, "r")
    plt.show()


def main():
    test_data = create_dataloader(
        63,
        "Data_train/Speech",
        "Data_train/Peaks",
        10,
        10,
        select=1)

    save_model = Saver('checkpoints/bdl_0babble')
    use_cuda = True

    model = MyNet()
    model, _, _ = save_model.load_checkpoint(
        model, file_name="bce_epoch_10.pt")

    test(model, test_data, use_cuda=use_cuda)


class Saver:
    def __init__(self, directory: str = 'pytorch_model',
                 iteration: int = 0) -> None:
        self.directory = directory
        self.iteration = iteration

    def load_checkpoint(self,
                        model: nn.Module,
                        optimizer: optim.Optimizer = None,
                        file_name: str = 'pytorch_model.pt'):
        filepath = os.path.join(self.directory, file_name)
        checkpoint = th.load(filepath)
        model.load_state_dict(checkpoint['model_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        hyperparam_dict = {
            k: v
            for k, v in checkpoint.items()
            if k != 'model_dict' or k != 'optimizer_dict'
        }

        return model, optimizer, hyperparam_dict


if __name__ == "__main__":
    main()