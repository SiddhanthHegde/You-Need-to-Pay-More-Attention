import os
from shutil import copy2
import time
import torch

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

def epoch_time(start_time,end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time/60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins,elapsed_secs

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for idx, data in enumerate(data_loader):

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)
        labelsviewed = labels.view(labels.shape[0],1)
        image = data['image'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            img=image
            )
        preds = [0 if x < 0.5 else 1 for x in outputs]
        preds = torch.tensor(preds).to(device)
        loss = loss_fn(outputs,labelsviewed)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      labels = d["label"].to(device)
      labelsviewed = labels.view(labels.shape[0],1)
      image = d['image'].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        img=image
      )
      preds = [0 if x < 0.5 else 1 for x in outputs]
      preds = torch.tensor(preds).to(device)
      loss = loss_fn(outputs, labelsviewed)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model,data_loader, device):
    model = model.eval()
    f_preds = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            image = d['image'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                img=image
            )
            preds = [0 if x < 0.5 else 1 for x in outputs]
            for j in preds:
                f_preds.append(j)
    
    return f_preds