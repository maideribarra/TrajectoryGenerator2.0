import numpy as np
import torch
import pandas as pd
import time, random, math, string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
import datetime
import numpy as np
import pandas as pd
import json    
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import datasets
from torch.utils.data import Dataset
import datetime
import numpy as np
import pandas as pd
import json    
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import datasets
from torch.utils.data import Dataset

class DataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, batch_size, seq_len):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.seq_len=seq_len


    def load_dataset(self, path):
        dfarr = pd.read_csv(path) 
        NParr = dfarr.to_numpy()
        sp=np.array_split(NParr, np.unique(NParr[:, 0], return_index=True)[1][1:])
        print('sp[0]',len(sp[0]))
        print('sp[0][0]',len(sp[0][0]))
        print('sp[1]',len(sp[1]))
        mediaSizeTraj=sum([len(x) for x in sp])/len(sp)
        arrDist=np.array([len(x) for x in sp])
        unique, counts = np.unique(arrDist, return_counts=True)
        print('media trayectorias',mediaSizeTraj)
        print('distribuciones',dict(zip(unique, counts)))
        print('media trayectorias',mediaSizeTraj)
        ## Filtrar trayectorias de 1000 puntos, ya que solo son 5
        max_len = self.seq_len
        arr = [np.pad(x, ((0, max_len - len(x)),(0,0)), 'constant', constant_values=0) for x in sp if len(x)<max_len]
        NParr=np.asarray(arr, dtype=float)[1:100, :, 1:]
        print(NParr.shape)
        tensorArr=torch.Tensor(NParr)
        dataset = tensorArr
        print(type(tensorArr))   
        print(type(NParr)) 
        print(type(dataset)) 
        return dataset

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        dataset = self.load_dataset(self.train_path)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def val_dataloader(self):
        dataset = self.load_dataset(self.val_path)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def test_dataloader(self):
        dataset = self.load_dataset(self.test_path)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
    
    def predict_dataloader(self):
        dataset = self.load_dataset(self.test_path)
        return torch.utils.data.DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False)