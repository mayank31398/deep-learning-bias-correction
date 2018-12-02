import warnings
import os
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, TensorDataset
import math
from time import localtime, strftime
from mnist import MNIST
import matplotlib.pyplot as plt

# np.random.seed(42)
# th.manual_seed(42)
# th.cuda.manual_seed_all(42)
# random.seed(42)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))

    def forward(self, x):
        y = self.c1(x)
        y = y.reshape(-1, 28, 28)

        return y


def create_dataloader(batch_size, path):
    data = MNIST("GAN/Data")

    x_train, y_train = data.load_training()
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test, y_test = data.load_testing()
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = x_train / 255
    x_test = x_test / 255

    dataset_train = TensorDataset(
        th.from_numpy(x_train.astype(np.float32)),
        th.from_numpy(y_train.astype(np.float32)))
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True)

    dataset_test = TensorDataset(
        th.from_numpy(x_test.astype(np.float32)),
        th.from_numpy(y_test.astype(np.float32)))
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True)

    return dataloader_train, dataloader_test


def test(model_G: nn.Module,
         use_cuda: bool = False):
    model_G.eval()

    if use_cuda:
        if th.cuda.is_available():
            model_G.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')

    z = Variable(th.randn(1, 1)).cuda()
    image = model_G(z)
    image = image.cpu().detach().numpy().reshape(28, 28)

    plt.imshow(image)
    plt.show()


def main():
    model_G = Generator()
    save_model = Saver('GAN/checkpoints')
    model_G, _, _ = save_model.load_checkpoint(model_G, file_name="bce_epoch_30.pt")

    test(model_G, use_cuda=True)


class Saver:
    def __init__(self, directory: str = 'pytorch_model',
                 iteration: int = 0) -> None:
        self.directory = directory
        self.iteration = iteration

    def save_checkpoint(self,
                        state,
                        file_name: str = 'pytorch_model.pt',
                        append_time=True):
        os.makedirs(self.directory, exist_ok=True)
        timestamp = strftime("%Y_%m_%d__%H_%M_%S", localtime())
        filebasename, fileext = file_name.split('.')
        if append_time:
            filepath = os.path.join(
                self.directory,
                '_'.join([filebasename, '.'.join([timestamp, fileext])]))
        else:
            filepath = os.path.join(self.directory, file_name)
        if isinstance(state, nn.Module):
            checkpoint = {'model_dict': state.state_dict()}
            th.save(checkpoint, filepath)
        elif isinstance(state, dict):
            th.save(state, filepath)
        else:
            raise TypeError('state must be a nn.Module or dict')

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

    def create_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                          hyperparam_dict):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()

        state_dict = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer_dict,
            'timestamp': strftime('%I:%M%p GMT%z on %b %d, %Y', localtime())
        }
        checkpoint = {**state_dict, **hyperparam_dict}

        return checkpoint


if __name__ == "__main__":
    main()
