#This file creates a balanced split between the classes and makes it model feedable form
import os
import pandas as pd
from zipfile import ZipFile
from utils import move_data, split_data

extract_path = 'training_img.zip'
with ZipFile(extract_path, 'r') as zipObj:
   zipObj.extractall()

os.mkdir('Troll')
os.mkdir('Non_troll')
src = 'uploaded_tamil_memes'
move_data(src,'Troll','Non_troll')

os.mkdir('Train')
os.mkdir('Val')
split_data('Troll','Train','Val',128)
split_data('Non_troll','Train','Val',101)

df = pd.read_csv('train_captions.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

train_df_data = []
val_df_data = []
for img_name in os.listdir('Train'):
  ind = list(df[df['imagename'] == img_name].index)[0]
  train_df_data.append([img_name,df['captions'].iloc[ind]])

for img_name in os.listdir('Val'):
  ind = list(df[df['imagename'] == img_name].index)[0]
  val_df_data.append([img_name,df['captions'].iloc[ind]])

train_df = pd.DataFrame(train_df_data,columns=['img_name','captions'])
val_df = pd.DataFrame(val_df_data,columns=['img_name','captions'])