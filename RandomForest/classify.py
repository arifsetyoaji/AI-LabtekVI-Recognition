import argparse, textwrap
import os

import h5py
import mahotas
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
fixed_size = (500,500)
test_size = 0.10
seed      = 12
bins = 8
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"
train_labels = ['Labtek_VI', 'not-Labtek_VI']

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# Fitting Model
model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
model.fit(global_features, global_labels)

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


def main():
    print('[INFO] Processing...')

    # Argument Parser ---------------------------
    parser = argparse.ArgumentParser(
        usage='classify.py [nama-file]', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'file',
        help=textwrap.dedent('''\
            File Pada Folder Lain -> classify.py [nama-folder]/[nama-file]
            Supported File Types -> jpg, jpeg
             '''))

    args = parser.parse_args()

    if os.path.isfile(args.file):
        if ((args.file[-3:] == 'jpg') or (args.file[-4:] == 'jpeg')):
            print('[INFO] Processed File : ', args.file, '\r\n')
            filename = args.file
        else:
            raise ValueError('Extensi File Tidak Didukung !')
    else:
        raise ValueError('File Tidak Ada !')

    # --------------------------------------------

    # read the image
    image = cv2.imread(filename)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    result = model.predict(global_feature.reshape(1,-1))[0]

    prob = model.predict_proba(global_feature.reshape(1,-1))

    print('------------------------------------------------ \r\n')

    print('[RESULT] Probabilitas Labtek VI : ', prob[0,0])

    if result == 0:
        print('[RESULT] {} is Labtek VI'.format(filename))
    elif result == 1:
        print('[RESULT] {} is NOT Labtek VI'.format(filename))

    cv2.putText(image, train_labels[result], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    print('\r\n------------------------------------------------')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
