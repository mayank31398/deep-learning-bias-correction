import warnings
import os
import numpy as np
import torch as th
import torch.nn as nn
import pandas as pd
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
            nn.Conv2d(8, 8, kernel_size = 5, padding = 2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size = 5, padding = 2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size = 5, padding = 2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.c2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size = 3, padding = 1), nn.BatchNorm2d(64), nn.Tanh())
        
        self.c3 = nn.Sequential(
            nn.Linear(3136, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 10), nn.BatchNorm1d(10)
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
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.c1(x)
        x = self.c2(x)
        x = x.reshape(x.shape[0], -1)
        y = self.c3(x)

        return y
    
def create_dataloader(batch_size, path):
    data = MNIST("Data")

    x_train, y_train = data.load_training()
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test, y_test = data.load_testing()
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    kaggle_x_test = pd.read_csv("Data/x_test.csv", index_col = None).values[:, 1:]
    kaggle_y_test = pd.read_csv("Data/y_test.csv", index_col = None).values[:, 1]

    x_train = x_train / 255
    x_test = x_test / 255
    kaggle_x_test = kaggle_x_test / 255

    dataset_train = TensorDataset(
        th.from_numpy(x_train.astype(np.float32)),
        th.from_numpy(y_train.astype(np.float32)))
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False)
    
    dataset_test = TensorDataset(
        th.from_numpy(x_test.astype(np.float32)),
        th.from_numpy(y_test.astype(np.float32)))
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False)
    
    dataset_kaggle = TensorDataset(
        th.from_numpy(kaggle_x_test.astype(np.float32)),
        th.from_numpy(kaggle_y_test.astype(np.float32)))
    dataloader_kaggle = DataLoader(
        dataset_kaggle,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False)
    
    return dataloader_train, dataloader_test, dataloader_kaggle
    
def test(model: nn.Module,
         test_loader: DataLoader,
         use_cuda: bool = False,
         threshold: float = 0.5):
    model.eval()
    loss_sum = 0
    batches = len(test_loader)
    correct = 0
    total = 0

    results = []
    if use_cuda:
        if th.cuda.is_available():
            model.cuda()
        else:
            print('Warning: GPU not available, Running on CPU')
    for x_test, y_test in test_loader:
        if use_cuda:
            y_test = y_test.type(th.LongTensor)
            x_test, y_test = x_test.cuda(), y_test.cuda()

        x_test, y_test = Variable(x_test), Variable(y_test)

        probabilities = model(x_test)
        net_loss = F.cross_entropy(probabilities, y_test)

        predictions = probabilities.argmax(dim = 1)
        correct += (predictions == y_test).sum().item()
        total += y_test.shape[0]

        results.append(predictions.cpu().numpy())

        loss_sum += net_loss.item()
        del x_test, y_test, net_loss, probabilities, predictions
        th.cuda.empty_cache()
    print(
        'Loss {} Accuracy {}'.
        format(loss_sum / batches, 100 * correct / total))
    
    results = np.concatenate(results)
    data_save = pd.DataFrame(results)
    data_save.to_csv("Predictions.csv")

def main():
    _, _, kaggle_data = create_dataloader(4096, "Data")
    model = MyNet()
    use_cuda = True

    save_model = Saver('Best/checkpoints')
    model, _, _ = save_model.load_checkpoint(model, file_name = "bce_epoch_95.pt")
    
    print("Kaggle set:")
    test(model, kaggle_data, use_cuda)

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