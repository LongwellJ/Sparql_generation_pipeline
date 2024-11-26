from refined.inference.processor import Refined
import json
from tqdm import trange

refined = Refined.from_pretrained(model_name='wikipedia_model',entity_set="wikipedia")
print('model loaded')

datafilepath = f'../data/LCQUAD1/LCQUAD1_test_final.json'

with open(datafilepath, 'r') as f: 
    train_data = json.load(f)
    ids = list(map(lambda x: x['id'], train_data))
    query = list(map(lambda x: x['query'], train_data))
    question = list(map(lambda x: x['question'], train_data))
    answers = list(map(lambda x: x['answers'], train_data))
    triples = list(map(lambda x: x['triples'], train_data))
    new_triples = list(map(lambda x: x['new_triples'], train_data))
results = []
counter = 0
total = 0
for i in trange(len(query)):
    spans = refined.process_text(question[i])
    #print(spans)
    if spans != []:
        counter += 1
    spanslist=list(spans)
    entites = []
    for j in range(len(spanslist)):
        entites.append([str(spanslist[j].predicted_entity.wikipedia_entity_title).replace(' ', '_')])

    results.append({'id': ids[i], 'question': question[i], 'query': query[i], 'answers':answers[i], 'triples':triples[i], 'new_triples':new_triples[i], 'entites':entites})
    total+=1
print(counter, total, counter/total)

with open(datafilepath, 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, indent=4)