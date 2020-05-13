import os
import cv2
import glob
import time
import shutil
import random
import tensorflow as tf
import numpy as np
from PIL import Image
from model import *

# base directories
data_dir = 'data\\'
raw_dir = os.path.join(data_dir, 'raw\\')
rubbish_dir = os.path.join(raw_dir, 'rubbish\\')
train_dir = os.path.join(data_dir, 'train\\')
val_dir = os.path.join(data_dir, 'validation\\')
is_player_dir = os.path.join(train_dir, 'player\\')
is_other_dir = os.path.join(train_dir, 'other\\')
raw_player_dir = os.path.join(raw_dir, 'player\\')
raw_other_dir = os.path.join(raw_dir, 'other\\')

# data filtering directories
prob_90_player = 'data\\sorting\\prob_90_player\\'
prob_80_player = 'data\\sorting\\prob_80_player\\'
prob_70_player = 'data\\sorting\\prob_70_player\\'
prob_60_player = 'data\\sorting\\prob_60_player\\'
prob_50_player = 'data\\sorting\\prob_50_player\\'
prob_40_player = 'data\\sorting\\prob_40_player\\'
prob_30_player = 'data\\sorting\\prob_30_player\\'
prob_20_player = 'data\\sorting\\prob_20_player\\'
prob_10_player = 'data\\sorting\\prob_10_player\\'
prob_not_player = 'data\\sorting\\prob_not_player\\'


# model directory
model_path = 'models\\MobileNet100_epochs.2e-05_lr'

def human_in_the_loop_labeling(classes=['player', 'other']):
    # look at maybe directories and manually sort
    # save images again in train/validation directory

    dirs = [prob_90_player, prob_80_player, prob_70_player, prob_60_player, prob_60_player,
    prob_50_player, prob_40_player, prob_30_player, prob_20_player, prob_10_player, prob_not_player]

    for d in dirs:
        frames = glob.glob(d + '*.jpg')
        
        for f in frames:
            # rename the file when moving
            new_name = str(random.randint(0, 999999)) + '.jpg'

            img = cv2.imread(f)
            cv2.imshow(f, img)
            
            # keyboard input
            key = cv2.waitKey(0)

            # label image and move file
            if key == ord('b'):
                # label as other
                new_path = os.path.join(is_other_dir, new_name)
                shutil.move(f, new_path)
                print(f, ' labeled as other and moved to ', new_path)

            elif key == ord('v'):
                # label as player
                new_path = os.path.join(is_player_dir, new_name)
                shutil.move(f, new_path)
                print(f, ' labeled as player', new_path)
            
            elif key == ord('n'):
                # move to rubbish
                new_path = os.path.join(rubbish_dir, new_name)
                shutil.move(f, new_path)
                print(f, ' moved to rubbish ', new_path)
            
            elif key == ord('q'):
                break

            cv2.destroyAllWindows()

def label_with_model():
    # lower bound for model confidence to label as a human
    
    model = load_model(model_path)
    
    frames = glob.glob(raw_player_dir + '*.jpg')
    
    for f in frames:
        # load image
        img = Image.open(f)
        i = tf.keras.preprocessing.image.img_to_array(img)
        i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
        
        # make predictions
        pred = model.predict(np.array([i]))
        print(pred)
        
        # assign new random name
        img_name = str(random.randint(0, 999999)) + '.jpg'

        # filter into directories
        if pred > 0.9:
            img.save(prob_90_player + img_name)
        elif pred > 0.8:
            img.save(prob_80_player + img_name)
        elif pred > 0.7:
            img.save(prob_70_player + img_name)
        elif pred > 0.6:
            img.save(prob_60_player + img_name)
        elif pred > 0.5:
            img.save(prob_50_player + img_name)
        elif pred > 0.4:
            img.save(prob_40_player + img_name)
        elif pred > 0.3:
            img.save(prob_30_player + img_name)
        elif pred > 0.2:
            img.save(prob_20_player + img_name)
        elif pred > 0.1:
            img.save(prob_10_player + img_name)
        else:
            img.save(prob_not_player + img_name)
        
        os.remove(f)



def main():
    human_in_the_loop_labeling()
    return



if __name__=='__main__':
    main()