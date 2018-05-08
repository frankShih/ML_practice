
import pandas as pd
import numpy as np

def loadAndFormatData(filename):
    print("load and format data ...")
    data = pd.read_csv(filename, sep=", ", header=None)
    data.columns = ["x1", "x2", "x3", "x4","x5","x6","x7","x8","x9","x10","x11","x12"]
    # data.drop(["x3", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16"], axis=1, inplace=True)
    return data



dataLeft = loadAndFormatData('sonarLogL.txt')
dataLeft1 = loadAndFormatData('sonarLogL1.txt')
dataLeft2 = loadAndFormatData('sonarLogL2.txt')
dataRight = loadAndFormatData('sonarLogR.txt')
dataRight1 = loadAndFormatData('sonarLogR1.txt')
dataRight2 = loadAndFormatData('sonarLogR2.txt')
dataMid = loadAndFormatData('sonarLogM.txt')
dataMid1 = loadAndFormatData('sonarLogM1.txt')
dataMid2 = loadAndFormatData('sonarLogM2.txt')

dataLeft = pd.concat([dataLeft, dataLeft1, dataLeft2], axis=0)
dataRight = pd.concat([dataRight, dataRight1, dataRight2], axis=0)
dataMid = pd.concat([dataMid, dataMid1, dataMid2], axis=0)


def removeOutliers(dataFrame):
    print("remove outliers")
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


dataLeft = removeOutliers(dataLeft)
dataRight = removeOutliers(dataRight)
dataMid = removeOutliers(dataMid)


def movingAvg(dataSet, winSize=5):
    print("moving average")
    ma_data = dataSet.copy()
    for ind, col in enumerate(dataSet):
        ma_data[col] = dataSet[col].rolling(window=winSize).mean()

    ma_data.dropna(inplace=True)
    ma_data.reset_index(drop=True, inplace=True)
    # print(ma_data.head(10), ma_data.shape)    
    return ma_data


dataLeft = movingAvg(dataLeft, winSize = 5)
dataRight = movingAvg(dataRight, winSize = 5)
dataMid = movingAvg(dataMid, winSize = 5)


def featureExtaction(df):
	print("extract features")
    result, header = [], []
        
    for i in range(df.shape[0]):
        temp = []
        for j in range(len(df.iloc[i])-1):
            for k in range(j+1, len(df.iloc[i])):
                temp.append(df.iloc[i, j] / df.iloc[i, k])
        result.append(temp)    
    
    for ind in range(len(temp)):
        header.append("x{}".format(1+ind))
    
    return pd.DataFrame(result, columns=header)


dataLeft = featureExtaction(dataLeft)
dataRight = featureExtaction(dataRight)
dataMid = featureExtaction(dataMid)



def labelAndCombineData(df_list):
    print("labelAndCombineData")
    data_list = []
    label_list = []
    for ind, df in enumerate(df_list):
        temp = df.copy()
        label = pd.Series(ind, index=df.index, dtype=int)
        data_list.append(temp)
        label_list.append(label)
    
    return pd.concat(data_list, axis=0), pd.concat(label_list, axis=0)


com_data, com_label = labelAndCombineData([dataLeft, dataRight, dataMid])



def normalizeTrainDF(dataFrame, mode="std"):
    print("normalize train df")
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


def normalizeTestDF(dataFrame, params, mode="std"):
    print("normalize test df")
    result = dataFrame.copy()    
    
    for ind, feature_name in enumerate(dataFrame.columns):        
        #print(ind, feature_name, dataFrame[feature_name], params["mean"][ind])
        if mode == "std":
            result[feature_name] = ((dataFrame[feature_name] - params["mean"][ind]) / params["std"][ind]) if params["std"][ind] else 0
        elif mode == "mean":
            result[feature_name] = ((dataFrame[feature_name] - params["mean"][ind]) / (params["max"][ind] - params["mean"][ind])) if (params["max"][ind] - params["mean"][ind]) else 0
        else:
            result[feature_name] = ((dataFrame[feature_name] - params["min"][ind]) / (params["max"][ind] - params["min"][ind])) if (params["max"][ind] - params["min"][ind]) else 0

    return result



from sklearn import linear_model
from sklearn import metrics, cross_validation
print("start training")
logreg = linear_model.LogisticRegression(C=1e-1)
logreg.fit(norm_data, com_label)
# predicted = cross_validation.cross_val_predict(logreg, norm_data, com_label, cv=100)
print("waiting for data input ...")

def makePrediction(dirname, filename):
    print("making prediction")
    open(dirname+'prediction.lock', 'a').close()
    try:
        data = loadAndFormatData(dirname+filename)
        data = removeOutliers(data)
        data = movingAvg(data,  winSize = 5)
        data = featureExtaction(data)        
        dataNorm = normalizeTestDF(data, params)
        with open(dirname+"prediction.txt", "a") as myfile:
            for i in range(dataNorm.shape[0]):
                current = dataNorm.iloc[i].reshape(1, -1)
                print(logreg.predict(current)[0])
                myfile.write(str(logreg.predict(current)[0]))
    except (OSError, IOError, pd.io.common.EmptyDataError) as e:
        print(e)
    finally:
        #os.remove('/sdcard/DCIM/logs/prediction.txt')
        if os.path.exists(dirname+filename):
            os.remove(dirname+filename)
        if os.path.exists(dirname+'prediction.lock'):
            os.remove(dirname+'prediction.lock')
    

import os.path
import time

while True:
    # print(os.path.exists('/sdcard/DCIM/logs/sonarLog.txt'), os.path.exists('/sdcard/DCIM/logs/sonarLog.lock'))
    if os.path.exists('/sdcard/DCIM/logs/sonarLog.txt') and not(os.path.exists('/sdcard/DCIM/logs/sonarLog.lock')):
        makePrediction('/sdcard/DCIM/logs/', 'sonarLog.txt')
        time.sleep(0.1)
'''    
        
# dimension reduction
from sklearn.decomposition import PCA 
pca=PCA(n_components=3, copy=True, whiten=False)
reduced_data = pca.fit_transform(norm_data)
print(pca.components_, pca.n_components_, pca.explained_variance_ratio_, pca.mean_, pca.noise_variance_)



import matplotlib.pyplot as plt
import ipympl
from mpl_toolkits.mplot3d import Axes3D




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(reduced_data[com_label==0, 0], reduced_data[com_label==0, 1], reduced_data[com_label==0, 2:], c='r', marker='o')
ax.scatter(reduced_data[com_label==1, 0], reduced_data[com_label==1, 1], reduced_data[com_label==1, 2:], c='b', marker='o')
ax.scatter(reduced_data[com_label==2, 0], reduced_data[com_label==2, 1], reduced_data[com_label==2, 2:], c='g', marker='o')
# ax.scatter(reduced_data[com_label==3, 0], reduced_data[com_label==3, 1], reduced_data[com_label==3, 2:], c='c', marker='o')
# ax.scatter(reduced_data[com_label==4, 0], reduced_data[com_label==4, 1], reduced_data[com_label==4, 2:], c='m', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
'''