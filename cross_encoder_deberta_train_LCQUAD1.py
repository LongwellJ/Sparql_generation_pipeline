from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DebertaTokenizer
from transformers import AutoTokenizer, DebertaForSequenceClassification

import torch
from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split
import json
from tqdm import trange
from tqdm import tqdm
import sys
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

weight_for_0 = 1 / (4857201 / (4857201 + 451198))
weight_for_1 = 1 / (451198 / (4857201 + 451198))

weights = torch.tensor([weight_for_1], dtype=torch.float32)
weights = weights.cuda()
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaForSequenceClassification.from_pretrained(f'./cross_encoder_one_class_deberta_LCQUAD1_2')
loss_fn = BCEWithLogitsLoss(pos_weight=weights)
train_filepath = f'../data/LCQUAD1/cross_encoder_LCQUAD1_train_filtered.json'
val_filepath = f'../data/LCQUAD1/cross_encoder_LCQUAD1_test_filtered.json'
with open(train_filepath, 'r') as file:
    data = json.load(file)
    input = list(map(lambda x: x['data'], data))
    target = list(map(lambda x: x['label'], data))

with open(val_filepath, 'r') as file:
    valdata = json.load(file)
    valinput = list(map(lambda x: x['data'], valdata))
    valtarget = list(map(lambda x: x['label'], valdata))
 
print('*'*10)
print('we have started tokenizing')
print('*'*10)
inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
valinputs = tokenizer(valinput, return_tensors="pt", padding=True, truncation=True)
print('*'*10)
print('we have finished tokenizing')
print('*'*10)
labels = torch.tensor(target, dtype=torch.float32)
vallabels = torch.tensor(valtarget, dtype=torch.float32)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'].float(), labels)
valdataset = TensorDataset(valinputs['input_ids'], valinputs['attention_mask'].float(), vallabels)
#torch.save(dataset, '../data/qald-9-plus/dataset.pt')
#torch.save(valdataset, '../data/qald-9-plus/valdataset.pt')

#dataset = torch.load('../data/qald-9-plus/dataset.pt')
#valdataset = torch.load('../data/qald-9-plus/valdataset.pt')
#train_set, val_set = train_test_split(dataset, test_size=0.05)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(valdataset, batch_size=512)

selected_gpu = sys.argv[1]
device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
for epoch in range(3):
    model.train()
    for input_ids, attention_mask, labels in tqdm(train_loader):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Evaluation
    eval_loss = 0
    eval_steps = 0
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss
            eval_loss += val_loss.item()
            eval_steps += 1
            #print(val_loss.item())
    print(eval_loss/eval_steps)
    # Save model
    model.save_pretrained(f'./cross_encoder_one_class_deberta_LCQUAD1_{epoch}')
    print("Model has been saved.")




