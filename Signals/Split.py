import numpy as np
from glob import glob
import os
import random

pathegg = 'Data/Peaks'
eggfiles = sorted(glob(os.path.join(pathegg, '*.npy')))

pathspeech = 'Data/Speech'
speechfiles = sorted(glob(os.path.join(pathspeech, '*.npy')))

shuffler = np.arange(len(speechfiles))
np.random.shuffle(shuffler)

train_indexes = shuffler[:27]
test_indexes = shuffler[27:]

eggpath_save = 'Data_train/Peaks'
speechpath_save = 'Data_train/Speech'
for i in train_indexes:
    file = speechfiles[i]
    basename = os.path.basename(file)
    data = np.load(file)
    np.save(speechpath_save + "/" + basename, data)
    
    file = eggfiles[i]
    basename = os.path.basename(file)
    data = np.load(file)
    np.save(eggpath_save + "/" + basename, data)

eggpath_save = 'Data_test/Peaks'
speechpath_save = 'Data_test/Speech'
for i in test_indexes:
    file = speechfiles[i]
    basename = os.path.basename(file)
    data = np.load(file)
    np.save(speechpath_save + "/" + basename, data)
    
    file = eggfiles[i]
    basename = os.path.basename(file)
    data = np.load(file)
    np.save(eggpath_save + "/" + basename, data)