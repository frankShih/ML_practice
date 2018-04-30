
import pandas as pd
import numpy as np

def loadAndFormatData(filename):
    data = pd.read_csv(filename, sep=", ", header=None)
    data.columns = ["x1", "x2", "x3", "x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x16"]
    data.drop(["x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16"], axis=1, inplace=True)
    return data


dataLeftMost = loadAndFormatData('sonarLogLeftMost.txt')
dataLeft = loadAndFormatData('sonarLogLeft.txt')
dataRightMost = loadAndFormatData('sonarLogRightMost.txt')
dataRight = loadAndFormatData('sonarLogRight.txt')
dataMid = loadAndFormatData('sonarLogMid.txt')



def removeOutliers(dataFrame):
    d_mean = dataFrame.mean()
    d_std = dataFrame.std()
    outlier_index = []
    for column in dataFrame:
        for ind, val in enumerate(dataFrame[column]):
            if val > (d_mean[column] + 3*d_std[column]) or val < (d_mean[column] - 3*d_std[column]):
                outlier_index.append(ind)

    outlier_index = list(set(outlier_index))
    result = dataFrame.drop(dataFrame.index[outlier_index])
    result.reset_index(drop=True, inplace=True)
    return result


dataLeftMost = removeOutliers(dataLeftMost)
dataLeft = removeOutliers(dataLeft)
dataRightMost = removeOutliers(dataRightMost)
dataRight = removeOutliers(dataRight)
dataMid = removeOutliers(dataMid)



def movingAvg(dataSet, winSize=5):
    ma_data = dataSet.copy()
    for ind, col in enumerate(dataSet):
        ma_data[col] = dataSet[col].rolling(window=winSize).mean()

    ma_data.dropna(inplace=True)
    ma_data.reset_index(drop=True, inplace=True)
    print(ma_data.head(10), ma_data.shape)    
    return ma_data


dataLeftMostMA = movingAvg(dataLeftMost,  winSize = 5)
dataLeftMA = movingAvg(dataLeft, winSize = 5)
dataRightMostMA = movingAvg(dataRightMost, winSize = 5)
dataRightMA = movingAvg(dataRight, winSize = 5)
dataMidMA = movingAvg(dataMid, winSize = 5)



def labelAndCombineData(df_list):
    data_list = []
    label_list = []
    for ind, df in enumerate(df_list):
        temp = df.copy()
        label = pd.Series(ind, index=df.index, dtype=int)
        data_list.append(temp)
        label_list.append(label)
    
    return pd.concat(data_list, axis=0), pd.concat(label_list, axis=0)

com_data, com_label = labelAndCombineData([pd.concat([dataLeftMostMA, dataLeftMA], axis=0), pd.concat([dataRightMostMA, dataRightMA], axis=0), dataMid])



# feature normalization
def normalizeTrainDF(dataFrame, mode="std"):
    result = dataFrame.copy()    
    params = pd.DataFrame(index=range(len(dataFrame.columns)),columns = ["std", "mean", "min", "max"])
    
    for ind, feature_name in enumerate(dataFrame.columns):
        std_value = dataFrame[feature_name].std()
        mean_value = dataFrame[feature_name].mean()
        max_value = dataFrame[feature_name].max()
        min_value = dataFrame[feature_name].min()
        params.iloc[ind] = [std_value, mean_value, max_value, min_value]        
        if mode == "std":
            result[feature_name] = ((dataFrame[feature_name] - mean_value) / std_value) if std_value else 0
        elif mode == "mean":
            result[feature_name] = ((dataFrame[feature_name] - mean_value) / (max_value - mean_value)) if (max_value - mean_value) else 0
        else:
            result[feature_name] = ((dataFrame[feature_name] - min_value) / (max_value - min_value)) if (max_value - min_value) else 0

    return result, params
        
norm_data, params = normalizeTrainDF(com_data)
print(norm_data.head(), params["std"][0])


def normalizeTestDF(dataFrame, params, mode="std"):
    result = dataFrame.copy()    
    
    for ind, feature_name in enumerate(dataFrame.columns):        
        if mode == "std":
            result[feature_name] = ((dataFrame[feature_name] - params["mean"][ind]) / params["std"][ind]) if params["std"][ind] else 0
        elif mode == "mean":
            result[feature_name] = ((dataFrame[feature_name] - params["mean"][ind]) / (params["max"][ind] - params["mean"][ind])) if (params["max"][ind] - params["mean"][ind]) else 0
        else:
            result[feature_name] = ((dataFrame[feature_name] - params["min"][ind]) / (params["max"][ind] - params["min"][ind])) if (params["max"][ind] - params["min"][ind]) else 0

    return result


from sklearn import linear_model
from sklearn import metrics, cross_validation

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(norm_data, com_label)
predicted = cross_validation.cross_val_predict(logreg, norm_data, com_label, cv=10)


def makePredction(filename):
    data = loadAndFormatData(filename)    
    dataMA = movingAvg(data,  winSize = 5)
    dataNorm = normalizeTestDF(dataMA, params)
    with open("/sdcard/DCIM/logs/prediction.txt", "a") as myfile:
        for i in range(dataNorm.shape[0]):
            current = dataNorm.iloc[i].reshape(1, -1)
#             print(current, str(logreg.predict(current)[0]))
            myfile.write(str(logreg.predict(current)[0]))
    


import os.path
import time

while True:
    if not os.path.exists('/sdcard/DCIM/logs/sonarLog.txt'):
        print("file not found ~~~")
        time.sleep(0.1)
    else:
        makePredction('/sdcard/DCIM/logs/sonarLog.txt')
        os.remove('/sdcard/DCIM/logs/sonarLog.txt')


