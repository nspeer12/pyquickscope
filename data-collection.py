import cv2
import numpy
from PIL import Image
import pyautogui
import glob
from vision import get_roi, screenshot
import matplotlib.pyplot as plt
from pynput import keyboard
from pynput.keyboard import Key, Controller
from pynput.mouse import Listener
import random
import time

window = [0, 0, 1920, 1080]
n = 300
roi = get_roi(window, n)

other_dir = "data\\raw\\other\\"
human_dir = "data\\raw\\player\\"

def on_press(key):
    img_name = str(random.randint(0, 999999)) + '.jpg'

    try:
        if key.char == 'x':
            exit(0)
        if key.char == 'q':
            print('person')
            img = screenshot(roi)
            img.save(human_dir + img_name)
        if key.char == 'shift':
            print('other')
            img = screenshot(roi)
            img.save(other_dir + img_name)

    except AttributeError:
        return

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False



def on_click(x, y, button, pressed):
    '''
    img_name = str(random.randint(0, 999999)) + '.jpg'
    if pressed:
        img = screenshot(roi)
        img.save(human_dir + img_name)
        print('player caputered: ', img_name)
    '''
    return

def on_scroll(x, y, button, pressed):
    img_name = str(random.randint(0, 999999)) + '.jpg'
    if pressed:
        img = screenshot(roi)
        img.save(other_dir + img_name)
        print('player caputered: ', img_name)

if __name__=='__main__':
    # get back into the game
    for i in range(5):
        print('waiting for ', 5 - i, ' seconds')
        time.sleep(1)

    # define window and region
    
    # create keyboard listener
    with Listener(on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join() 