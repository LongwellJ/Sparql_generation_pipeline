import json
from tqdm import tqdm
import kneed
train_filepath = f'../data/LCQUAD1/LCQUAD1_test_finalwith_retrieved_triples.json'
triples_filepath = f'../data/LCQUAD1/cross_encoder_LCQUAD1_test_filtered_with_scores.json'
with open(train_filepath, 'r') as file:
    data = json.load(file)
    ids = list(map(lambda x: x['id'], data))
    query = list(map(lambda x: x['query'], data))
    questions = list(map(lambda x: x['question'], data))
    answers = list(map(lambda x: x['answers'], data))
    #new_answers = list(map(lambda x: x['new_answers'], data))
    new_triples = list(map(lambda x: x['new_triples'], data))
    triples = list(map(lambda x: x['triples'], data))
    entities = list(map(lambda x: x['entities'], data))

with open(triples_filepath, 'r') as file2:
    data2 = json.load(file2)
    id2 = list(map(lambda x: x['id'], data2))
    questions2 = list(map(lambda x: x['question'], data2))
    triples2 = list(map(lambda x: x['triples'], data2))
    score = list(map(lambda x: x['score'], data2))
    labels = list(map(lambda x: x['label'], data2))
# Create dataset
step = 0
dataset = []
for current_id in tqdm(ids):
    final_triples = []
    score_list = []
    true_labels = []
    for i in range(len(questions2)):
        if current_id == id2[i] and float(score[i]) < 7:
            #print("This one works",list(triples2[i][:3]))
            true_label = labels[i]
            #print(true_label)

            final_triples.append(list(triples2[i][:3]))
            score_list.append(float(score[i]))
            true_labels.append(true_label)
    #print(final_triples, score_list)
    final_triples = [t for _, t in sorted(zip(score_list, final_triples), reverse=False)]
    true_labels = [t for _, t in sorted(zip(score_list, true_labels), reverse=False)]
    score_list = sorted(score_list, reverse=False)
    #print(final_triples, score_list)
    dataset.append({'id': ids[step], 'question': questions[step], 'query': query[step], 'answers':answers[step], 'entities': entities[step], 'gold_triples': triples[step], 'new_gold_triples': new_triples[step], 'retrieved_triples': final_triples, 'score': score_list, 'true_labels':true_labels})
    step += 1

with open(f'../data/LCQUAD1/LCQUAD1_test_ranked_triples_entities.json', 'w') as file3:
    json.dump(dataset, file3, indent=4)
print('done')


import numpy as np
def clean_data(prop):
    clean = prop.split('/')[-1].replace('>', '').replace('_', ' ')
    return clean

list_of_lengths = []
list_of_ones_per_question_top_5 = []

list_of_total_true_gold_triples = []
list_of_total_found_gold_triples = []
for dict in dataset:
    list_of_scores = dict['score']
    list_of_true_labels = dict['true_labels']
    gold_triples = dict['gold_triples']
    retrieved_triples = dict['retrieved_triples']
    list_of_lengths.append(len(list_of_scores))

    number_of_ones_per_question_top10 = 0
    if len(list_of_true_labels) >100:

        for i in list_of_true_labels[0:100]:
        #print(i)
            if i == 1:
                number_of_ones_per_question_top10 +=1
    else:
        for i in list_of_true_labels:
            if i == 1:
                number_of_ones_per_question_top10 +=1
    list_of_ones_per_question_top_5.append(number_of_ones_per_question_top10)

    # gold_triple_found = 0
    # total_true_gold_triples = 0
    # for z in range(len(gold_triples)):
    #     for j in gold_triples[z]:
    #         #print(j)
    #         total_true_gold_triples +=1
    #         for m in range(len(retrieved_triples)):
    #             #print('this is m ', m)
    #             matches = 0
    #             variables = 0
    #             for i in range(3):
    #                 try:
    #                     if clean_data(j[i]).startswith('?'):
    #                             variables+=1
    #                 except Exception as error:
    #                     print("An exception occurred:", error)
    #             for k in retrieved_triples[m]:
    #                 for i in range(3):
    #                     #print(j[i])
    #                     #print(k)
    #                     try:
    #                         if j[i]==k:
    #                             matches+=1
    #                     except Exception as error:
    #                         print("An exception occurred:", error)
    #             #print('variables', variables)
    #             #print('matches',matches)            
    #             if 3-variables <=matches:
    #                 gold_triple_found+=1
    #                 break
    #         # if skip ==True:
    #         #     break
    
    
    # list_of_total_found_gold_triples.append(gold_triple_found)
    # list_of_total_true_gold_triples.append(total_true_gold_triples)
        
                

                    

array_of_total_found_gold_triples = np.array(list_of_total_found_gold_triples)
array_of_total_true_gold_triples = np.array(list_of_total_true_gold_triples)

array_of_number_of_ones = np.array(list_of_ones_per_question_top_5)
array_of_lengths = np.array(list_of_lengths)

print(array_of_lengths)
print(sum(array_of_lengths))
print(sum(array_of_lengths>0))
print(array_of_number_of_ones)
print(sum(array_of_number_of_ones>0))
print(sum(array_of_number_of_ones)/1000)



# print(array_of_total_found_gold_triples)
# print(sum(array_of_total_found_gold_triples))
# print(sum(array_of_total_found_gold_triples>0))
# print(array_of_total_true_gold_triples)
# print(sum(array_of_total_true_gold_triples))

