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
import os
import win32api as wapi
from alexnet import alexnet

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
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


keyList=["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys=[]
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)

    return keys

WIDTH=80
HEIGHT=60
LR=1e-3
EPOCHS=8

MODEL_NAME='pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

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
    ReleaseKey(W)
    ReleaseKey(A)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


last_time = time.time()
paused=False

while True:

    if not paused:
        
         screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
         screen =  cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
         screen =  cv2.resize(screen,(80,60))

    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    

    prediction=model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]

    moves = list(np.around(prediction))

    print(moves,prediction)

    if moves == [1,0,0]:
        left()
    elif moves == [0,1,0]:
        straight()
    elif moves == [0,0,1]:
        right()

    keys=key_check()

    if 'T' in keys:
        if paused:
            paused=False
            time.sleep(1)
        else:
            paused=TRUE
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)         
                
main()
