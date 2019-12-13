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
from keras.models import load_model
from keras.preprocessing import image as preproc

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
fixed_size = (500,500)
bins = 8
train_labels = ['Labtek_VI', 'not-Labtek_VI']


# Fitting Model
model = load_model("cnn-labtek6.h5")

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

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

    image = preproc.load_img(filename, target_size=(64, 64))
    image = preproc.img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    result = model.predict_classes(image)[0,0]
    prob = model.predict_proba(image)

    print('------------------------------------------------ \r\n')

    print('[RESULT] Probabilitas Labtek VI : ',1-prob[0,0])

    if result == 0:
        print('[RESULT] {} is Labtek VI'.format(filename))
    elif result == 1:
        print('[RESULT] {} is NOT Labtek VI'.format(filename))


    image2 = cv2.imread(filename)
    image2 = cv2.resize(image2, (500,500))

    cv2.putText(image2, train_labels[result], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.show()

    print('\r\n------------------------------------------------')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
