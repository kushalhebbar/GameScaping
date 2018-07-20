import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_value=np.load('training_data_2.npy')
print(len(train_value))

df=pd.DataFrame(train_value)
print(df.head())
print(Counter(df[1].apply(str)))

move_lefts=[]
move_rights=[]
move_forwards=[]

shuffle(train_value)

for data in train_value:
    img=data[0]
    choice=data[1]

    if choice == [0,0,1]:
        move_rights.append([img, choice])
        
    elif choice == [0,1,0]:
        move_forwards.append([img, choice])

    elif choice == [1,0,0]:
        move_lefts.append([img, choice])

    else:
        print('No Matches!!!')


move_forwards=move_forwards[:len(move_lefts)][:len(move_rights)]

move_lefts=move_lefts[:len(move_forwards)]
move_rights=move_rights[:len(move_rights)]

final_value=move_forwards+move_lefts+move_rights

shuffle(final_value)
print(len(final_value))
np.save('training_data_v5.npy', final_value)
