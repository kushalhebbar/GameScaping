import numpy as np
from PIL import ImageGrab
import cv2
import time
#import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
#from directkeys import PressKey,ReleaseKey, W, A, S, D
from statistics import mean
import ctypes
import time


SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    valuepress = Input_I()
    valuepress.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), valuepress )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    valuerelease= Input_I()
    valuerelease.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), valuerelease )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)

    PressKey(0x1E)
    time.sleep(1)
    ReleaseKey(0x1E)
    time.sleep(1)

    PressKey(0x1F)
    time.sleep(1)
    ReleaseKey(0x1F)
    time.sleep(1)

    PressKey(0x20)
    time.sleep(1)
    ReleaseKey(0x20)
    time.sleep(1)

def roi(img, vertices):
    
    #blank mask:
    mask = np.zeros_like(img)   
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 255)
    
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_img(image):
    original_image = image
    # edge detection
    processed_image =  cv2.Canny(image, threshold1 = 200, threshold2=300)
    
    processed_image = cv2.GaussianBlur(processed_image,(5,5),0)
    
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)

    processed_image = roi(processed_image, [vertices])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:        
    lines = cv2.HoughLinesP(processed_image, 1, np.pi/180, 180,      20,       15)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1,m2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_image, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_image,original_image, m1, m2

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)


last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    new_screen,original_image, left_slope, right_slope = process_img(screen)
    cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))



    if left_slope < 0 and right_slope < 0:
        right()
    elif left_slope > 0  and right_slope > 0:
        left()
    else:
        straight()
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
