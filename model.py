import torch
import torch.nn as nn
import timm
from transformers import AutoModel 

class multimodal(nn.Module):
  def __init__(self):
    super(multimodal, self).__init__()
    self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
    self.vit.head = nn.Linear(self.vit.head.in_features, 128)
    self.fc1 = nn.Linear(self.bert.config.hidden_size,128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(256,1)
    self.drop = nn.Dropout(p=0.2)

  def forward(self,input_ids, attention_mask, img):
    _, pooled_output = self.bert(
      input_ids = input_ids,
      attention_mask = attention_mask
    )
    text_out = self.fc1(pooled_output)
    img_out = self.vit(img)
    merged = torch.cat((text_out,img_out),1)
    act = self.relu(merged)
    out = self.drop(act)
    return self.fc2(out)