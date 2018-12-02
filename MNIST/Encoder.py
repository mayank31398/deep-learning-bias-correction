import warnings
import os
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

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 5, padding = 2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size = 5, padding = 2), nn.BatchNorm2d(16), nn.ReLU()
        )
        
        self.c2 = nn.MaxPool2d(2, return_indices = True)

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.c4 = nn.MaxPool2d(2, return_indices = True)

        self.c5 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.Tanh())

        self.c6 = nn.MaxUnpool2d(2)

        self.c7 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size = 3, padding = 1), nn.BatchNorm2d(16), nn.ReLU()
        )

        self.c8 = nn.MaxUnpool2d(2)

        self.c9 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size = 5, padding = 2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size = 5, padding = 2), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        self._initialize_submodules()

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal(m.weight.data)
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.out_channels
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.c1(x)
        x, ind1 = self.c2(x)
        x = self.c3(x)
        x, ind2 = self.c4(x)
        x = self.c5(x)
        x = self.c6(x, ind2)
        x = self.c7(x)
        x = self.c8(x, ind1)
        y = self.c9(x)
        y = y.reshape(y.shape[0], -1)

        return y
    
def create_dataloader(batch_size, path):
    data = MNIST("Data")

    x_train, _ = data.load_training()
    x_train = np.array(x_train)

    x_test, _ = data.load_testing()
    x_test = np.array(x_test)

    x_train = x_train / 255
    x_test = x_test / 255

    dataset_train = TensorDataset(
        th.from_numpy(x_train.astype(np.float32))
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True)
    
    dataset_test = TensorDataset(
        th.from_numpy(x_test.astype(np.float32))
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True)
    
    return dataloader_train, dataloader_test

def train(model: nn.Module,
          optimizer: optim.Optimizer,
          train_data: DataLoader,
          use_cuda: bool = True,
          scheduler=None):
    model.train()
    loss_sum = 0
    batches = len(train_data)

    if use_cuda:
        if th.cuda.is_available():
            model.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    for x_train, in train_data:
        if scheduler is not None:
            scheduler.step()

        if use_cuda:
            x_train = x_train.cuda()

        x_train = Variable(x_train)
        optimizer.zero_grad()

        reconstructions = model(x_train)
        net_loss = ((reconstructions - x_train) ** 2).sum() / x_train.shape[0]

        loss_sum += net_loss.item()

        net_loss.backward()
        # TODO: Gradient Clipping
        optimizer.step()
    del net_loss
    th.cuda.empty_cache()

    return loss_sum / batches
    
def test(model: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool = False,
         threshold: float = 0.5):
    model.eval()
    loss_sum = 0
    batches = len(test_loader)

    if use_cuda:
        if th.cuda.is_available():
            model.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    for x_test, in test_loader:
        if use_cuda:
            x_test = x_test.cuda()

        x_test = Variable(x_test)

        reconstructions = model(x_test)
        net_loss = ((reconstructions - x_test) ** 2).sum() / x_test.shape[0]

        loss_sum += net_loss.item()
        del x_test, net_loss, reconstructions
        th.cuda.empty_cache()
    print(
        'Test Loss {}\n'.
        format(loss_sum / batches))

def main():
    train_data, test_data = create_dataloader(4096, "Data")

    model = MyNet()
    save_model = Saver('Encoder')

    use_cuda = True
    epochs = 1001

    optimizer = optim.Adamax(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.9)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i in range(epochs):
            nloss = train(model, optimizer, train_data, use_cuda)
            print(
                'Train Loss {} @epoch {}'.
                format(nloss, i))

            if(i % 5 == 0):
                test(model, test_data, use_cuda)
                checkpoint = save_model.create_checkpoint(
                    model, optimizer, {
                        'win': 160,
                        'stride': 5
                    })
                save_model.save_checkpoint(
                    checkpoint,
                    file_name='bce_epoch_{}.pt'.format(i),
                    append_time=False)
            if scheduler is not None:
                scheduler.step()

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