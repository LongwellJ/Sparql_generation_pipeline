from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import trange
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, DebertaForSequenceClassification

# Load tokenizer and model
#model = BertForSequenceClassification.from_pretrained('../scripts/cross_encoder_one_class',num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaForSequenceClassification.from_pretrained('../scripts/cross_encoder_one_class_deberta_LCQUAD1_0')



test_filepath = f'../data/LCQUAD1/cross_encoder_LCQUAD1_train_filtered.json'
with open(test_filepath, 'r') as file:
    data = json.load(file)
    ids = list(map(lambda x: x['id'], data))
    questions = list(map(lambda x: x['question'], data))
    input = list(map(lambda x: x['data'], data))
    target = list(map(lambda x: x['label'], data))
    triple = list(map(lambda x: x['triple'], data))

print('*'*10)
print('we have started tokenizing')
print('*'*10)
inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
print('*'*10)
print('we have finished tokenizing')
print('*'*10)
labels = torch.tensor(target)
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

#test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset, batch_size=512, shuffle=False)

selected_gpu = sys.argv[1]
device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Inference

ones = []
zeros = []
results = []
model.eval()
i = 0
with torch.no_grad():
    for input_ids, attention_mask, batch_labels in tqdm(test_loader):
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        batch_labels = batch_labels.to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().cpu().numpy()
        for logit, label in zip(logits, batch_labels):
            score = str(logit[0])
            if label.item() == 1:
                ones.append(score)
            elif label.item() == 0:
                zeros.append(score)
            
            results.append({'id': ids[i], 'question': questions[i], 'triples': triple[i], 'data': input[i], 'label': label.item(), 'score': score})
            i += 1


finished_filepath = '../data/LCQUAD1/cross_encoder_LCQUAD1_train_filtered_with_scores.json'
with open(finished_filepath, 'w') as file:
    json.dump(results, file, indent=4)

np.save(f'./logit_arrays/ones_cased_deberta_train_LCQUAD1.npy', ones)
np.save(f'./logit_arrays/zeros_cased_deberta_train_LCQUAD1.npy', zeros)

# plt.hist(ones, bins=10)
# plt.xlabel('logit')
# plt.ylabel('Frequency')
# plt.title('Histogram of logit for label=1')
# plt.show()
# plt.savefig('logit_label1_cased.png')

# plt.clf()

# plt.hist(zeros, bins=10)
# plt.xlabel('logit')
# plt.ylabel('Frequency')
# plt.title('Histogram of logit for label=0')
# plt.show()
# plt.savefig('logit_label0_cased.png')

# Calculate the percentage of numbers greater than 0.2

# greater_than_point_two = np.asarray(ones) > 0.2
# percentage = np.mean(greater_than_point_two) * 100

# # Print the result
# print(f"Percentage of ones greater than 0.2: {percentage}%")

# greater_than_point_two = np.asarray(zeros) > 0.2
# percentage = np.mean(greater_than_point_two) * 100

# # Print the result
# print(f"Percentage of zeros greater than 0.2: {percentage}%")

# try:
#     print('The average logit = ', total_logit/total)
# except:
#     print("the converted average logit = ", total_logit.float()/total)
# print(f"test 1 Accuracy: {correct_1 / total_1}")
# print(f"test 0 Accuracy: {correct_0 / total_0}")