import torch
from PIL import Image
from torch.utils.data import DataLoader

class TamilDataset(torch.utils.data.Dataset):
  def __init__(self,df,tokenizer,max_len,path,transforms=None):
    self.data_dir = path
    self.df = df
    self.tokenizer = tokenizer
    self.transforms = transforms
    self.max_len = max_len

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self,index):
    img_name, captions = self.df.iloc[index]
    img_path = os.path.join(self.data_dir,img_name)
    labels = 0 if img_name.startswith('N') else 1
    img = Image.open(img_path).convert('RGB')

    if self.transforms is not None:
      img = self.transforms(img)

    encoding = self.tokenizer.encode_plus(
        captions,
        add_special_tokens=True,
        max_length = self.max_len,
        return_token_type_ids = False,
        padding = 'max_length',
        return_attention_mask= True,
        return_tensors='pt',
        truncation=True
    )

    return {
        'image' : img,
        'text' : captions,
        'input_ids' : encoding['input_ids'].flatten(),
        'attention_mask' : encoding['attention_mask'].flatten(),
        'label' : torch.tensor(labels,dtype=torch.float)
    } 

def create_data_loader(df,tokenizer,max_len,batch_size,mytransforms,path,shuffle):
  ds = TamilDataset(
      df,
      tokenizer,
      max_len,
      path,
      mytransforms
  )

  return DataLoader(ds,
                    batch_size = batch_size,
                    shuffle=False,
                    num_workers=4)