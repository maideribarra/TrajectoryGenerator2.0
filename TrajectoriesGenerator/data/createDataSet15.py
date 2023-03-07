import random
import pandas as pd
import numpy
if __name__=='__main__':
    #ftrain = open("train.dat", "a")
    #ftest = open("test.dat", "a")
    #fval = open("val.dat", "a")
    fdataset = pd.read_csv("LunarLanderv3.dat")
    print(fdataset)
    #arrDset=fdataset.to_numpy()
    #arrDset= fdataset.read().split('\n')
    #print(arrDset)
    arrOrdDsetTrain=fdataset.iloc[1:707704,1:]
    arrOrdDsetTest=fdataset.iloc[707705:796164,1:]
    arrOrdDsetVal=fdataset.iloc[796165:884630,1:]
    
    #sep='\n'
    #arrOrdDsetTrainS=sep.join(arrOrdDsetTrain[:,1:])
    #arrOrdDsetTestS=sep.join(arrOrdDsetTest[:,1:])
    #arrOrdDsetValS=sep.join(arrOrdDsetVal[:,1:])
    print(arrOrdDsetTrain.shape)
    #ftrain.write(arrOrdDsetTrain)
    #ftest.write(arrOrdDsetTest)
    #fval.write(arrOrdDsetVal)
    #ftrain.close()
    #ftest.close()
    #fval.close()
    arrOrdDsetTrain.to_csv("train.dat", sep=',', index=False)
    arrOrdDsetTest.to_csv("test.dat", sep=',', index=False)
    arrOrdDsetVal.to_csv("val.dat", sep=',', index=False)

