import cv2
import sys
import time
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from PIL import ImageGrab


def get_roi(window, n):
    n = n/2
    # returns n by n region of interest in the center of the screen
    xdiff = window[2] - window[0]
    ydiff = window[3] - window[1]
    xcenter = xdiff / 2
    ycenter = ydiff / 2
    roi =  [int(xcenter - n), int(ycenter - n), int(xcenter + n), int(ycenter + n)]
    return roi


def screenshot(roi):
    return ImageGrab.grab(bbox=roi)


def fps(t1):
    t2 = time.time()
    f = 1 / (t2 - t1)
    sys.stdout.write("FPS  %.2f\r" % f)
    return t2

def record(roi, show=True, show_fps=True):
    record = True
    t = time.time()

    while(record):
        frame = np.array(screenshot(roi))
        if show:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        if show_fps:
            t = fps(t)


if __name__ == '__main__':
    roi = get_roi([0,0,1920,1080], 300)
    record(roi)