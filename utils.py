import os
from shutil import copy2

def move_data(start,troll,not_troll):
  for img_name in os.listdir(start):
    src = os.path.join(start,img_name)
    if img_name.startswith('N'):
      copy2(src,not_troll)
    else:
      copy2(src,troll)

def split_data(start,train,val,split):
  for i, img_name in enumerate(os.listdir(start)):
    src = os.path.join(start,img_name)
    if i < split:
      copy2(src,val)
    else:
      copy2(src,train)

