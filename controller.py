import pyautogui
import time
def click(x, y):
    pyautogui.click(x, y)

def get_position():
    return pyautogui.position()


def quickscope(x, y):
    # hold right for 1 second
    aim(1, x, y)
    pyautogui.click(button='left')
    #pyautogui.mouseUp()


def gratata(n):
    pyautogui.click(button='left', clicks=n, interval=0.1)


def full_auto(n, x, y):
    pyautogui.dragTo(x, y, n, button='left')


def aim(n, x, y):
    pyautogui.mouseDown(button='right',x=x, y=y, duration=n)


def drag_scope(n, x, y):
    aim(n,x,y)
    pyautogui.click()


def lock_on_target(objectRegion):
    xcenter = int((objectRegion[2] - objectRegion[0]) / 2) + objectRegion[0]
    ycenter = int((objectRegion[3] - objectRegion[1]) / 2) + objectRegion[1]

    # bad coordinates
    if ycenter < 0 or xcenter < 0:
        print('invalid coordinates')
        return
    else:
        x, y = get_position()
        pyautogui.move(x - xcenter, y - ycenter)
        