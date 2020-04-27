import cv2
import numpy
from PIL import Image
import pyautogui
import glob
from vision import get_roi, screenshot
import matplotlib.pyplot as plt
from pynput import keyboard
from pynput.keyboard import Key, Controller
import random

window = [0, 0, 1920, 1080]
n = 300
roi = get_roi(window, n)

other_dir = "data\\other\\"
human_dir = "data\\human\\"

def on_press(key):
    img_name = str(random.randint(0, 999999)) + '.jpg'

    try:
        if key.char == 'q':
            exit(0)
        if key.char == 'v':
            print('person')
            img = screenshot(roi)
            img.save(human_dir + img_name)
        if key.char == 'b':
            print('other')
            img = screenshot(roi)
            img.save(other_dir + img_name)

    except AttributeError:
        return

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


if __name__=='__main__':
    # define window and region
    
    
    # create keyboard listener
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
            listener.join()

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()