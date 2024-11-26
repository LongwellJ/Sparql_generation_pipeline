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
model = DebertaForSequenceClassification.from_pretrained('../scripts/cross_encoder_one_class_deberta')



test_filepath = f'../data/qald-9-plus/cross_encoder_QALD_testset_filtered.json'
with open(test_filepath, 'r') as file:
    data = json.load(file)
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

test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
selected_gpu = sys.argv[1]
device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Inference

correct_1 = 0
correct_0 = 0
total_1 = 0
total_0 = 0
total=0
total_logit=0
zeros = []
ones = []
results = []
model.eval()
i=0
with torch.no_grad():
    for input_ids, attention_mask, labels in tqdm(test_loader):
        label = 0
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logit = outputs.logits
        logit = logit.detach().cpu().numpy()
        #print(logit)
        #print(logit)
        #total_logit+=logit
        #total+=1
        #print(logit)
        # Use argmax to get the predicted label
        #predicted_labels = torch.argmax(logit, dim=1)
        #compare the predicted labels to the actual labels if the actual label is 1
        #print(predicted_labels)
        #print(labels)
        if labels == torch.tensor([1], device='cuda:0'):
            ones.append(logit[0][0])
            label=1
        #     #print(ones)
        #     # if labels == predicted_labels:
        #     #     correct_1 +=1
        #     # total_1 +=1

        if labels == torch.tensor([0], device='cuda:0'):
            zeros.append(logit[0][0])
            label=0
        # #     if labels == predicted_labels:
        # #         correct_0 +=1
        # #     total_0 +=1
        # #print(str(logit[0][0]))

        results.append({'question': questions[i],'triples': triple[i], 'data': input[i], 'label': label, 'score':str(logit[0][0])})
        i+=1


finished_filepath = f'../data/qald-9-plus/cross_encoder_QALD_testset_filtered_with_scores.json'
with open(finished_filepath, 'w') as file:
    json.dump(results, file, indent=4)
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

np.save('ones_cased_deberta_test.npy', ones)
np.save('zeros_cased_deberta_test.npy', zeros)

# try:
#     print('The average logit = ', total_logit/total)
# except:
#     print("the converted average logit = ", total_logit.float()/total)
# print(f"test 1 Accuracy: {correct_1 / total_1}")
# print(f"test 0 Accuracy: {correct_0 / total_0}")