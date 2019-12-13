# library: basics
import numpy as np
import matplotlib.pyplot as plt

# library: directory
import os

# library: OpenCV
import cv2

# library: svm
from sklearn import svm
from sklearn.model_selection import GridSearchCV

def ftrxtract(fpath):
    #print(path)
    img = cv2.imread(fpath)
    img = cv2.resize(img, (250, 250), interpolation = cv2.INTER_CUBIC)
    rgb_ftr = img.flatten()
    return rgb_ftr

def get_data_from(dpath):
    # define empty container lists
    dat = []
    test = []

    # extract features for every image in the directory
    for subdir, dirs, files in os.walk(dpath):
        for file in files:
            fpath = subdir + os.sep + file
            if fpath.endswith(".jpg"):
                if subdir.endswith("Labtek_VI"):
                    #print(fpath)
                    add = ftrxtract(fpath)
                    add = np.append(add, 1)
                    dat.append(add)
                elif subdir.endswith("Labtek_VI_not"):
                    #print(fpath)
                    add = ftrxtract(fpath)
                    add = np.append(add, 0)
                    dat.append(add)

    # convert dat list -> float array
    dat = np.asarray(dat)
    dat = dat.astype(float)
    
    # split
    x = dat[:, :-1]
    y = dat[:, -1]
    
    # return extracted features
    return(x, y)

#get features
x, y = get_data_from(os.getcwd())

#parameter tuning
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(x, y)