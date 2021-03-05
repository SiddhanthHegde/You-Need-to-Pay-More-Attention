import torch
import pandas as pd
from dataset import create_data_loader
from model import multimodal
from utils import get_predictions
from zipfile import ZipFile

LOAD_MODEL = True
device = 'cuda'

model = multimodal()
model = model.to(device)

if LOAD_MODEL:
    model.load_state_dict(torch.load('vit-bert-1.0val.bin'))

df_test = pd.read_csv('test_captions.csv')
df_test.drop('Unnamed: 0',axis=1,inplace=True)
extract_path = 'test_img.zip'
with ZipFile(extract_path, 'r') as zipObj:
   zipObj.extractall()

test_data_loader = create_data_loader(df_test,tokenizer,MAX_LEN,BATCH_SIZE,my_trans,'test_img',False)
submission_preds = get_predictions(model,test_data_loader,device)

