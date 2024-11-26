import json
from tqdm import tqdm
train_filepath = f'qald_9_plus_test_dbpedia_en_triples_correct_final.json'
triples_filepath = f'cross_encoder_QALD_testset_filtered_with_scores.json'
with open(train_filepath, 'r') as file:
    data = json.load(file)
    ids = list(map(lambda x: x['id'], data))
    query = list(map(lambda x: x['query'], data))
    questions = list(map(lambda x: x['question'], data))
    answers = list(map(lambda x: x['answers'], data))
    new_answers = list(map(lambda x: x['new_answers'], data))

with open(triples_filepath, 'r') as file:
    data2 = json.load(file)
    questions2 = list(map(lambda x: x['question'], data2))
    triples2 = list(map(lambda x: x['triples'], data2))
    score = list(map(lambda x: x['score'], data2))


# Create dataset
step = 0
dataset = []
for q in tqdm(questions):
    final_triples = []
    score_list = []
    for i in range(len(questions2)):
        if q == questions2[i] and float(score[i]) > 0.5:
            final_triples.append(list(triples2[i][:3]))
            score_list.append(float(score[i]))
    #print(final_triples, score_list)
    final_triples = [t for _, t in sorted(zip(score_list, final_triples), reverse=True)]
    score_list = sorted(score_list, reverse=True)
    #print(final_triples, score_list)
    dataset.append({'id': ids[step], 'question': q, 'query': query[step], 'answers':answers[step], 'new_answers':new_answers[step], 'retrieved_triples': final_triples, 'score': score_list})
    step += 1

with open('QALD_Mistral_test_retrieved_triples.json', 'w') as file:
    json.dump(dataset, file, indent=4)
print('done')




