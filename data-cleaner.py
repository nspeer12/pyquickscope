import os
import cv2
import glob
import time
import shutil
import random

# base directories
data_dir = 'data\\'
raw_dir = os.path.join(data_dir, 'raw\\')
rubbish_dir = os.path.join(raw_dir, 'rubbish\\')
train_dir = os.path.join(data_dir, 'train\\')
val_dir = os.path.join(data_dir, 'validation\\')
is_player_dir = os.path.join(train_dir, 'player\\')
is_other_dir = os.path.join(train_dir, 'other\\')

# data filtering directories
raw_player_dir = os.path.join(raw_dir, 'player\\')
raw_other_dir = os.path.join(raw_dir, 'other\\')
maybe_player_dir = os.path.join(raw_player_dir, 'suspected\\')
maybe_not_player_dir = os.path.join(raw_player_dir, 'unsuspected\\')
maybe_other_dir = os.path.join(raw_other_dir, 'suspected\\')
maybe_not_other_dir = os.path.join(raw_other_dir, 'unsuspected\\')


def human_in_the_loop_labeling(classes=['player', 'other']):
    # look at maybe directories and manually sort
    # save images again in train/validation directory
    frames = glob.glob(raw_player_dir + '*.jpg')
    
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

def main():
    human_in_the_loop_labeling()

    return



if __name__=='__main__':
    main()