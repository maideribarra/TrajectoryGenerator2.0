import random
if __name__=='__main__':
    ftrain = open("train.dat", "a")
    ftest = open("test.dat", "a")
    fval = open("val.dat", "a")
    fdataset = open("LunarLanderv2_Obs.csv", "r")
    arrDset= fdataset.read().split('\n')
    print(arrDset)
    arrOrdDsetTrain=arrDset[1:298882]
    arrOrdDsetTest=arrDset[298883:336975]
    arrOrdDsetVal=arrDset[336976:373745]
    sep='\n'
    arrOrdDsetTrainS=sep.join(arrOrdDsetTrain)
    arrOrdDsetTestS=sep.join(arrOrdDsetTest)
    arrOrdDsetValS=sep.join(arrOrdDsetVal)
    ftrain.write(arrOrdDsetTrainS)
    ftest.write(arrOrdDsetTestS)
    fval.write(arrOrdDsetValS)
    ftrain.close()
    ftest.close()
    fval.close()


