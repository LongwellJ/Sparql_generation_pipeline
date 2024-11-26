from refined.inference.processor import Refined
import json
from tqdm import trange

refined = Refined.from_pretrained(model_name='wikipedia_model',entity_set="wikipedia")
print('model loaded')

datafilepath = f'../data/lc-quad-2/train.json'

with open(datafilepath, 'r') as f: 
    train_data = json.load(f)
    # ids = list(map(lambda x: x['id'], train_data))
    # query = list(map(lambda x: x['query'], train_data))
    question = list(map(lambda x: x['question'], train_data))
    # answers = list(map(lambda x: x['answers'], train_data))
    # triples = list(map(lambda x: x['triples'], train_data))
    # new_triples = list(map(lambda x: x['new_triples'], train_data))
results = []
counter = 0
total = 0
errors = 0
for i in trange(len(question)):
    try:
        spans = refined.process_text(question[i])
        #print(spans)
        if spans != []:
            counter += 1
        #results.append({'id': ids[i], 'question': question[i], 'query': query, 'answers':answers[i], 'triples':triples[i], 'new_triples':new_triples[i] })
        total+=1
    except:
        print('there was an error')
        errors+=1
print(counter, total, counter/total, errors)

# with open('datafilepath', 'w', encoding='utf-8') as outfile:
#     json.dump(results, outfile, indent=4)