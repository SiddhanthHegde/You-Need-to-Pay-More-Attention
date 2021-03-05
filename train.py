import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import timm
from zipfile import ZipFile
import os
import time
from shutil import copy2
from torch.utils.data import DataLoader
from transformers import AdamW,get_linear_schedule_with_warmup,AutoModel,AutoTokenizer
from PIL import Image
from collections import defaultdict

from model import multimodal
from dataset import create_data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

my_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),       
])

BATCH_SIZE = 16
MAX_LEN = 128
EPOCHS = 4
history = defaultdict(list)
best_accuracy = 0
LOAD_MODEL = False

train_data_loader = create_data_loader(train_df,tokenizer,MAX_LEN,BATCH_SIZE,my_trans,'Train',True)
val_data_loader = create_data_loader(val_df,tokenizer,MAX_LEN,BATCH_SIZE,my_trans,'Val',False)

model = multimodal()
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader)  * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss = nn.BCEWithLogitsLoss().to(device)

for epoch in range(EPOCHS):
 
 
  start_time = time.time()
  train_acc,train_loss = train_epoch(
      model,
      train_data_loader,
      loss,
      optimizer,
      device,
      scheduler,
      2071
  )
   
  
  val_acc,val_loss = eval_model(
      model,
      val_data_loader,
      loss,
      device,
      229
  )
  
  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'Train Loss {train_loss} accuracy {train_acc}')
  print(f'Val Loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

if history['val_acc'][-1] > 0.95:
    torch.save('vit-bert-1.0val.bin')

if LOAD_MODEL:
    model.load_state_dict(torch.load('vit-bert-1.0val.bin'))