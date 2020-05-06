import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt
from model import *
import glob

model_file = 'models\\Xception100-epochs.2e-05.lr_32'
classes_dir = 'data\\test\\*'

classes = {1: 'player', 2:'other'}

def main():
    # load model
    model = load_model(model_file)

    # make predictions
    classes = glob.glob(classes_dir)
    
    for c in classes:
        imgs = glob.glob(c + str('\\*.jpg'))
        for img in imgs:
            i = tf.keras.preprocessing.image.load_img(img, target_size=(300,300))
            i = tf.keras.preprocessing.image.img_to_array(i)
            i = tf.keras.applications.xception.preprocess_input(i)
            print(model.predict(np.array([i])))      
            frame = cv2.imread(img)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()