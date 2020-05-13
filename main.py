import numpy as np
import cv2
import sys
import time
import random
import matplotlib.pyplot as plt
from model import *
from vision import *
from controller import *
import glob

model_file = 'models\\MobileNet.250_epochs.2e-05_lr'
classes_dir = 'data\\test\\*'

raw_player = 'data\\raw\\player\\'
raw_other = 'data\\raw\\other\\'


classes = {1: 'player', 2:'other'}

window = [0, 0, 1920, 1080]
n = 300
roi = get_roi(window, n)



def detect_images():
    print('loading model ', model_file, '. This may take a minute.')

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



def screen_detection(confidence=0.7):
    print('loading model ', model_file, '. This may take a minute.')
    model = load_model(model_file)
    
    record = True
    while record:
        t1 = time.time()

	    # screen capture and processing
        img = screenshot(roi)
        #i = tf.keras.preprocessing.image.load_img(img, target_size=(300,300))
        i = tf.keras.preprocessing.image.img_to_array(img)
        i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
        
        
        
        # make predictions
        pred = model.predict(np.array([i]))
        
        t2 = time.time()
        
        pred_class = 'player' if pred > confidence else 'other'
        print(pred_class)

        # display image
        cv2.imshow('frame', np.array(img))
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            record = False


def first_person_shooter(confidence=0.5, mode=1):
    print('loading model ', model_file, '. This may take a minute.')
    model = load_model(model_file)
    
    play = True
    while play:
        t1 = time.time()

	    # screen capture and processing
        img = screenshot(roi)
        img_name = str(random.randint(0, 999999)) + '.jpg'

        #i = tf.keras.preprocessing.image.load_img(img, target_size=(300,300))
        i = tf.keras.preprocessing.image.img_to_array(img)
        i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
        
        # make predictions
        pred = model.predict(np.array([i]))
        pred_class = 'player' if pred > confidence else 'other'

        if pred_class == 'player':
            if mode == 1:
                quickscope(0,0)
            elif mode == 2:
                gratata(1)
            elif mode == 3:
                gratata(3)
            
            # save captured data
            img.save(raw_player + img_name)


        # fps counter
        t2 = time.time()
        print('Prediction: ', pred_class, pred, '\tFPS: ', 1/(t2-t1))

        # display image
        cv2.imshow('frame', np.array(img))
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            record = False

def main():
    first_person_shooter(mode=1)


if __name__ == '__main__':
    main()