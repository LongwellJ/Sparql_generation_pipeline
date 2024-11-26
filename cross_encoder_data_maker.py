from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
from tqdm import trange
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.functional import cosine_similarity

# Load tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


# # Function to get the BERT embeddings for a given text
# def get_bert_embedding(sentence):
#     # Add special tokens adds [CLS] and [SEP] tokens
#     input_ids = tokenizer.encode(sentence, add_special_tokens=True)
#     input_ids = torch.tensor([input_ids])  # Convert to tensor

#     with torch.no_grad():
#         outputs = model(input_ids)
#         hidden_states = outputs[2]

#     # Use the average of the last 4 layers.
#     token_vecs = hidden_states[-4:]
#     sentence_embedding = torch.mean(torch.stack(token_vecs), 0).squeeze()
#     # Use the mean of the sentence embeddings as the final sentence vector
#     return torch.mean(sentence_embedding, dim=0)

# # Compute the cosine similarity between two sentences
# def bert_cosine_similarity(embedding1, embedding2):
#     # Calculate cosine similarity
#     cos_sim = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
#     return cos_sim.item()

#cleaning function
def clean_data(prop):
    clean = prop.split('/')[-1].replace('>', '').replace('_', ' ')
    return clean


datapath = 'LCQUAD1_train_finalwith_retrieved_triples.json'
with open(datapath, 'r') as file:
    data = json.load(file)
    ids = list(map(lambda x: x['id'], data))
    query = list(map(lambda x: x['query'], data))
    questions = list(map(lambda x: x['question'], data))
    answers = list(map(lambda x: x['answers'], data))
    #new_answers = list(map(lambda x: x['new_answers'], data))
    triples = list(map(lambda x: x['triples'], data))
    new_triples = list(map(lambda x: x['new_triples'], data))
    entities = list(map(lambda x: x['entities'], data))
    retrieved_triples = list(map(lambda x: x['retreived_triples'], data))

# Create dataset
#vectorizer = TfidfVectorizer()
dataset = []
total_gold = 0
total_noise = 0
check_gold = 0
empty = 0
#go through each question in the dataset
for i in trange(len(questions)): 
    gold = False
    #question_embedding = get_bert_embedding(questions[i])
    #go through each retrieved triple dimension for the question
    for z in range(len(retrieved_triples[i])):
        #print("this is z",z)
        #print(len(retrieved_triples[i][z]))
        #go through each triple in the retrieved triples dimension
        for j in retrieved_triples[i][z]:
            #set skip to false
            skip = False
            #go through each new_triples dimension in the question
            for m in range(len(triples[i])):
                #print("this is m",m)
                #break out if skip is true
                #go through each triple in the new_triples dimension
                if skip == True:
                    break
                for k in triples[i][m]:
                    #print("this is k",k)
                    #check if elements in the retrieved triple are in the gold triple
                    variables = 0
                    matches = 0
                    for p in range(3):
                        if clean_data(k[p]).startswith('?'):
                            variables += 1
                        if clean_data(j[p]) == clean_data(k[p]):
                            matches += 1
                    if matches >= 3-variables:
                        j.append(1)
                        skip = True
                        total_gold += 1
                        gold = True
                        break    
                    # try:
                    #     if (j[0]==k[0] and j[1]==k[1]) or (j[2]==k[2] and j[1]==k[1]) or (j[0]==k[0] and j[2]==k[2]):
                    #         #if they are, append a relevance value of 1 and set skip to true
                    #         j.append(1)
                    #         skip = True
                    #         total_gold += 1
                    #         gold = True
                    #         #tfidf_matrix = vectorizer.fit_transform([questions[i], f'{clean_data(j[1])}'])
                    #         #cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                    #         #triple_embedding = get_bert_embedding(f'{clean_data(j[1])}')
                    #         #similarity = bert_cosine_similarity(question_embedding, triple_embedding)

                    #         #print("this is j",j)
                    #         break
                    # except:
                    #     #print(k)
                    #     print('there was an error, most likely the gold triple is empty')#if (j[0]==k[0] and j[1]==k[1]) or (j[2]==k[2] and j[1]==k[1]) or (j[0]==k[0] and j[2]==k[2]): IndexError: list index out of range
            if skip == False:
                #if skip is false, append a relevance value of 0        
                j.append(0)
                total_noise += 1
                #triple_embedding = get_bert_embedding(f'{clean_data(j[1])}')
                #similarity = bert_cosine_similarity(question_embedding, triple_embedding)
                #tfidf_matrix = vectorizer.fit_transform([questions[i], f'{clean_data(j[1])}'])
                #cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            dataset.append({'id':ids[i] ,'question': questions[i], 'triple': j, 'data':f'question: {questions[i]} [SEP] triple: {clean_data(j[0])} {clean_data(j[1])} {clean_data(j[2])}', 'label':j[3]})
    if answers[i] == {}:
        empty += 1
    if answers[i] != {} and gold ==True:
        check_gold += 1


print(check_gold/(len(questions)-empty))
print(total_gold)
print(total_noise)
with open('cross_encoder_LCQUAD1_train_filtered.json', 'w') as f:
    json.dump(dataset, f, indent=4)

# # Compute cosine similarity between questions and j
# question_embeddings = model.get_input_embeddings()(tokenizer(questions, return_tensors="pt", padding=True, truncation=True)["input_ids"])
# j_embeddings = model.get_input_embeddings()(tokenizer([j[0] for j in dataset], return_tensors="pt", padding=True, truncation=True)["input_ids"])
# similarity_scores = cosine_similarity(question_embeddings, j_embeddings)

# # Print similarity scores
# for i, question in enumerate(questions):
#     print(f"Similarity scores for question: {question}")
#     for j_index, j in enumerate(dataset):
#         print(f"Triple: {j['triple']}, Similarity score: {similarity_scores[i][j_index]}")
#     print()


# with open('cross_encoder_QALD.json', 'r') as file:
#     data = json.load(file)
#     input = list(map(lambda x: x['data'], data))
#     target = list(map(lambda x: x['label'], data))

# inputs = tokenizer(input[0:10], return_tensors="pt", padding=True, truncation=True)
# labels = torch.tensor(target[0:10])
# dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
# train_set, val_set = train_test_split(dataset, test_size=0.2)
# train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=16)

# # Training loop

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# for epoch in trange(3):
#     model.train()
#     for input_ids, attention_mask, labels in train_loader:
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         for input_ids, attention_mask, labels in val_loader:
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             val_loss = outputs.loss
#             print(val_loss.item())

# # Save model
# model.save_pretrained('./my_trained_model')

# # Load model
# model = BertForSequenceClassification.from_pretrained('./my_trained_model')
# tokenizer = BertTokenizer.from_pretrained('./my_trained_model')

# #Inference
# correct = 0
# total = 0
# model.eval()
# with torch.no_grad():
#     for input_ids, attention_mask, labels in train_loader:
#         input_ids = input_ids.to(model.device)
#         attention_mask = attention_mask.to(model.device)
#         labels = labels.to(model.device)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         logit = outputs.logits
#         # Use argmax to get the predicted label
#         predicted_labels = torch.argmax(logit, dim=1)
#         if labels == 1:
#             if predicted_labels == labels:
#                 correct += 1
#             total += 1

# # #compare the predicted labels with the actual labels
# # for i in range(len(predicted_labels)):
# #     if labels[i] == 1:
            
# #         if predicted_labels[i] == labels[i]:
# #             correct += 1
# #         total += 1        



