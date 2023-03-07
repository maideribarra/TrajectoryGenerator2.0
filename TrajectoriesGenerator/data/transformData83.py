import numpy as np
import torch
import pandas as pd

def transformData(rutaCsv):
    dfarr = pd.read_csv(rutaCsv) 
    NParr = dfarr.to_numpy()
    sp=np.array_split(NParr, np.unique(NParr[:, 0], return_index=True)[1][1:])
    print('NParr',NParr)
    print('NParr[0]',NParr[0])
    print('sp[0]',len(sp[0]))
    print('sp[0][0]',len(sp[0][0]))
    print('sp[1]',len(sp[1]))
    mediaSizeTraj=sum([len(x) for x in sp])/len(sp)
    arrDist=np.array([len(x) for x in sp])
    unique, counts = np.unique(arrDist, return_counts=True)
    print('media trayectorias',mediaSizeTraj)
    print('distribuciones',dict(zip(unique, counts)))
    print('media trayectorias',mediaSizeTraj)
    print('distribuciones',dict(zip(unique, counts)))
    ## Filtrar trayectorias de 1000 puntos, ya que solo son 5
    max_len = max([len(x) for x in sp])
    max_len = 137
    arr = [np.pad(x, ((0, max_len - len(x)),(0,0)), 'edge') for x in sp if len(x)<max_len]
    print('arrTrain[0][0]',arr[0][0])
    print('arrTrain[0]',arr[0])
    NParr=np.asarray(arr, dtype=float)[:, :, 1:]
    print(NParr.shape)
    print(NParr[0][0])
    Tarr=torch.from_numpy(NParr)
    return Tarr

