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

np.random.seed(42)
th.manual_seed(42)
th.cuda.manual_seed_all(42)
random.seed(42)


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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
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
        x = x.reshape(-1, 784)
        y = self.c1(x)
        
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


def train(model_G: nn.Module,
          model_D: nn.Module,
          optimizer_G: optim.Optimizer,
          optimizer_D: optim.Optimizer,
          train_data: DataLoader,
          use_cuda: bool = True):
    model_G.train()
    model_D.train()
    loss_sum = 0
    loss_D = 0
    loss_G = 0
    D_real_prob = 0
    D_fake_prob = 0
    batches = len(train_data)

    if use_cuda:
        if th.cuda.is_available():
            model_G.cuda()
            model_D.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    
    for x_train, y_train in train_data:
        if use_cuda:
            y_train = y_train.type(th.LongTensor).cuda()
            x_train, y_train = x_train.cuda(), y_train.cuda()
        
        batch_size = x_train.shape[0]

        x_train, y_train = Variable(x_train), Variable(y_train)
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Training the DISCRIMINATOR
        z = Variable(th.randn(batch_size, 1)).cuda()
        ones_label = Variable(th.ones(batch_size, 1)).cuda()
        zeros_label = Variable(th.zeros(batch_size, 1)).cuda()
        image = model_G(z)

        D_real = model_D(x_train)
        D_fake = model_D(image)

        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake
        
        D_real_prob += D_real.mean().item()
        D_fake_prob += D_fake.mean().item()

        D_loss.backward()
        optimizer_D.step()
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        
        # Training the GENERATOR
        for i in range(10):
            z = Variable(th.randn(batch_size, 1)).cuda()
            image = model_G(z)
            D_fake = model_D(image)
            G_loss = F.binary_cross_entropy(D_fake, ones_label)

            G_loss.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

        loss_D += D_loss.item()
        loss_G += G_loss.item()
        loss_sum += loss_D + loss_G
    th.cuda.empty_cache()

    return loss_sum / batches, loss_D / batches, loss_G / batches, D_real_prob / batches, D_fake_prob / batches


def test(model_G: nn.Module,
         model_D: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool = False,
         threshold: float = 0.5):
    model_G.eval()
    model_D.eval()
    loss_sum = 0
    loss_D = 0
    loss_G = 0
    D_real_prob = 0
    D_fake_prob = 0
    batches = len(test_loader)

    if use_cuda:
        if th.cuda.is_available():
            model_G.cuda()
            model_D.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    
    for x_test, y_test in test_loader:
        if use_cuda:
            y_test = y_test.type(th.LongTensor).cuda()
            x_test, y_test = x_test.cuda(), y_test.cuda()
        
        batch_size = x_test.shape[0]

        x_test, y_test = Variable(x_test), Variable(y_test)

        # Training the DISCRIMINATOR
        z = Variable(th.randn(batch_size, 1)).cuda()
        ones_label = Variable(th.ones(batch_size, 1)).cuda()
        zeros_label = Variable(th.zeros(batch_size, 1)).cuda()
        image = model_G(z)

        D_real = model_D(x_test)
        D_fake = model_D(image)

        D_loss_real = F.binary_cross_entropy(D_real, ones_label)
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake
        
        D_real_prob += D_real.mean().item()
        D_fake_prob += D_fake.mean().item()
        
        # Training the GENERATOR
        z = Variable(th.randn(batch_size, 1)).cuda()
        image = model_G(z)
        D_fake = model_D(image)
        G_loss = F.binary_cross_entropy(D_fake, ones_label)

        loss_D += D_loss.item()
        loss_G += G_loss.item()
        loss_sum += loss_D + loss_G
    th.cuda.empty_cache()
    print(
        'Loss {} D_loss {} G_loss {} D {} G {}\n'.
        format(loss_sum / batches, loss_D / batches, loss_G / batches, D_real_prob / batches, D_fake_prob / batches))


def main():
    train_data, test_data = create_dataloader(1024, "Data")

    model_G = Generator()
    model_D = Discriminator()
    save_model = Saver('GAN/checkpoints')

    use_cuda = True
    epochs = 1001

    optimizer_G = optim.Adamax(model_G.parameters(), lr=2e-3)
    optimizer_D = optim.Adamax(model_D.parameters(), lr=2e-3)
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, 10, 0.9)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, 10, 0.9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in range(epochs):
            loss_sum, loss_D, loss_G, D_real_prob, D_fake_prob = train(model_G, model_D, optimizer_G, optimizer_D, train_data, use_cuda)
            print(
                'Loss {} D_loss {} G_loss {} D {} G {} @epoch {}'.
                format(loss_sum, loss_D, loss_G, D_real_prob, D_fake_prob, i))

            if(i % 1 == 0):
                print("\nTest set")
                test(model_G, model_D, test_data, use_cuda)

                checkpoint = save_model.create_checkpoint(
                    model_G, optimizer_G, {
                        'win': 160,
                        'stride': 5
                    })
                save_model.save_checkpoint(
                    checkpoint,
                    file_name='bce_epoch_{}.pt'.format(i),
                    append_time=False)
            if scheduler_G is not None:
                scheduler_G.step()
                scheduler_D.step()


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