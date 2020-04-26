import numpy as np
import cv2
from mouse import *
import sys
import time
from vision import Vision
import threading
from queue import Queue
import matplotlib.pyplot as plt



def main(num_jobs, num_workers):
    
    printFps = True
    printStatus = False

    for x in range(num_workers):
        t = threading.Thread(target = threader)
        t.daemon = True
        t.start()

    for worker in range(num_jobs):
        q.put(worker)

    q.join()

    '''
    while(True):

        img, box = vision.detection()
        cv2.imshow('detection', img)

        if vision.shoot_status() == True:
            quickscope(None, None)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    '''


def job(worker):
    vision.detection()
    if vision.shoot_status() == True:
        gratata(3)
    #vision.print_info()


def threader():
    while(True):
        worker = q.get()
        job(worker)
        q.task_done()
    

def get_roi(window, n):
    n = n/2
    # returns n by n region of interest in the center of the screen
    xdiff = window[2] - window[0]
    ydiff = window[3] - window[1]
    xcenter = xdiff / 2
    ycenter = ydiff / 2
    roi =  [int(xcenter - n), int(ycenter - n), int(xcenter + n), int(ycenter + n)]
    return roi


def get_coords(window, objectRegion, n):
    n = n/2
    # returns the real location of the boxes relative to the gameplay screen
    x1 = window[0]
    y1 = window[1]
    x2 = window[2]
    y2 = window[3]

    roixorgin = int(((x2 - x1) / 2) - n)
    roiyorgin = int(((y2 - y1) / 2) - n)

    objectRegion = [objectRegion[0] + roixorgin,
                    objectRegion[1] + roiyorgin,
                    objectRegion[2] + roixorgin,
                    objectRegion[3] + roiyorgin]

    return objectRegion


if __name__ == '__main__':
    q = Queue()
    #vis_lock = threading.Lock()

    # x1, y1, x2, y2
    window = [0, 0, 1920, 1080]

    # n defines the height and width of our square viewing region
    n = 300
    roi = get_roi(window, n)
    vision = Vision(window, roi, n)
    
    num_jobs = 1000
    num_workers = 1
    main(num_jobs, num_workers)



def fps_test():
    q = Queue()
    #vis_lock = threading.Lock()

    # x1, y1, x2, y2
    window = [0, 0, 1920, 1080]

    # n defines the height and width of our square viewing region
    n = 300
    roi = get_roi(window, n)
    vision = Vision(window, roi, n)

    worker_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    avg_fps = []

    num_jobs = 100
    for w in worker_nums:
        start = time.time()
        main(num_jobs, w)
        #print('Average fps: ', num_jobs / (time.time()-start))
        avg_fps.append(num_jobs / (time.time()-start))
    
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(len(worker_nums)):
        ax.scatter(worker_nums[i], avg_fps[i])
    
    plt.show()