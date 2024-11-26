from refined.inference.processor import Refined
import json
from tqdm import trange

refined = Refined.from_pretrained(model_name='wikipedia_model',entity_set="wikipedia")
print('model loaded')

datafilepath = f'../data/qald-9-plus/qald_9_plus_test_dbpedia_en_triples_correct_final.json'

with open(datafilepath, 'r') as f: 
    train_data = json.load(f)
    ids = list(map(lambda x: x['id'], train_data))
    query = list(map(lambda x: x['query'], train_data))
    question = list(map(lambda x: x['question'], train_data))
    answers = list(map(lambda x: x['answers'], train_data))
    new_answers = list(map(lambda x: x['new_answers'], train_data))
    triples = list(map(lambda x: x['triples'], train_data))
    new_triples = list(map(lambda x: x['new_triples'], train_data))
results = []
counter = 0
total = 0
foo = 0
for i in trange(len(query)):
    spans = refined.process_text(question[i])
    #print(spans)
    
    spanslist=list(spans)
    #print(type(spanslist))
    entites = []
    for j in range(len(spanslist)):
        # print(type(spanslist))
        # print(type(str(spanslist[j])))
        # print(str(spanslist[j]))
        # print(spanslist[j].candidate_entities)
        # print(str(spanslist[j].predicted_entity.wikipedia_entity_title))
        
        # print(str(spanslist[j].predicted_entity.wikipedia_entity_title).replace(' ', '_'))
        # print(str(spanslist[j].candidate_entities))
        # print(spanslist[j])
        if str(spanslist[j].predicted_entity.wikipedia_entity_title).replace(' ', '_') != "None":

            entites.append([str(spanslist[j].predicted_entity.wikipedia_entity_title).replace(' ', '_')])
        # print(entites)
    results.append({'id': ids[i], 'question': question[i], 'query': query[i], 'answers':answers[i], 'new_answers':new_answers[i], 'triples':triples[i], 'new_triples':new_triples[i], 'entites':entites})
    if entites != []:
        counter += 1
    total+=1
print(counter, total, counter/total)
#print(results)
with open(f'../data/qald-9-plus/qald_9_plus_test_dbpedia_en_triples_correct_final_with_entities.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, indent=4)


