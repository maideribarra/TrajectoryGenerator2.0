import random
import pandas as pd
import numpy as np

def partitionateDataset(fileOrigin, fileTrain, fileTest, fileVal, index1, index2, index3):
    fdataset = pd.read_csv(fileOrigin)
    print(fdataset)
    arrOrdDsetTrain=fdataset.iloc[1:index1,1:]
    arrOrdDsetTest=fdataset.iloc[index1+1:index2,1:]
    arrOrdDsetVal=fdataset.iloc[index2+1:index3,1:]
    print(arrOrdDsetTrain.shape)
    arrOrdDsetTrain.to_csv(fileTrain, sep=',', index=False)
    arrOrdDsetTest.to_csv(fileTest, sep=',', index=False)
    arrOrdDsetVal.to_csv(fileVal, sep=',', index=False)

def oneHotEncodingActions(fileO,fileD):
    arr = np.loadtxt(fileO, delimiter=",")[1:,:]
    print(arr)
    print('file dimensions',arr.shape)
    lenDim = len(arr[0])
    print('line dimensions',lenDim)
    arrActions=arr[:,lenDim-1].astype(int)
    print('arrActions ',arrActions.shape)
    print('arrActions type ',type(arrActions))
    print('arrActions type 0',type(arrActions[0]))
    print('max arrActions',np.max(arrActions)+1)
    print(arrActions)
    encoded_array = np.zeros((arrActions.size, np.max(arrActions)+1), dtype=int)
    encoded_array[np.arange(arrActions.size),arrActions] = 1
    print('encoded array',encoded_array)
    print('encoded array shape',encoded_array.shape)
    print('shape file without actions',arr[:,:lenDim-1].shape)
    arrResult = np.concatenate((arr[:,:lenDim-1], encoded_array), axis=1)
    print('arrayResult shape', arrResult.shape)
    print('arrResult index 0',arrResult[0])
    np.savetxt(fileD,arrResult, delimiter=',')
    




if __name__=='__main__':
    #oneHotEncodingActions('/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/train40000.dat','/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/train40000v2.dat')
    #oneHotEncodingActions('/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/test40000.dat','/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/test40000v2.dat')
    oneHotEncodingActions('/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/val40000.dat','/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/val40000v2.dat')