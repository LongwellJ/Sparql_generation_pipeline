# Filter English questions

import json
import sys

# include_all_langs = False
#dataset_type = 'train' if sys.argv[1] == 'train' else 'test'
# initial_path = f'../data/qald-9-plus/qald_9_plus_{dataset_type}_dbpedia.json'
# final_path = f'../data/qald-9-plus/qald_9_plus_{dataset_type}_dbpedia_en.json'
# final_contents = []
# with open(initial_path, 'r') as f:
#   test_data = json.load(f)['questions']
#   for idx in range(len(test_data)):
#     test_sample = test_data[idx]
#     final_content = {}
#     final_content['id'] = test_sample["id"]
#     for elem in test_sample["question"]:
#       for k, v in elem.items():
#         if k == 'language':
#           assert v == 'en'
#         elif k == 'string':
#           final_content['question'] = v
#       if not include_all_langs:
#         break
#     for k, v in test_sample["query"].items():
#       if k == 'sparql':
#         final_content['query'] = v
#     final_content['answers'] = []
#     assert len(test_sample['answers']) == 1
#     answers = test_sample['answers'][0]
#     if 'results' in answers.keys():
#       bindings = answers['results']['bindings']
#       final_content['answers'].extend(bindings)
#     elif 'boolean' in answers.keys():
#       final_content['answers'].extend([answers['boolean']])
#     final_contents.append(final_content)
# with open(final_path, 'w') as f:
#   json.dump(final_contents, f, indent=4)

# Parse SPARQL

import re
from tqdm import trange
from copy import deepcopy

def extract_triples(inp):
  print(list(map(lambda elem: list(map(lambda item: \
                                             item.strip().split(' '), \
                                             elem.strip().split(' ; '))), inp.split('. '))))
  print('tada')
  return list(map(lambda elem: list(map(lambda item: \
                                             item.strip().split(' '), \
                                             elem.strip().split(' ; '))), inp.split('. ')))
def escape_special_characters(pattern):
    special_characters = r"?"
    escaped_pattern = re.sub(f"[{''.join(re.escape(c) for c in special_characters)}]", r"\\\g<0>", pattern)
    return escaped_pattern

def simplify_query(inp, prefixes, exceptions):
  # replace each prefix with its corresponding value in each triple
  try:
    simplified_query = re.sub(r'\s*(\w+):(\w+)\s+', lambda m: f' {prefixes[m.group(1)]}:{m.group(2)} ', inp)
  except:
    simplified_query = re.sub(r'\s+(\w+):(\w+)\s*', lambda m: f' {prefixes[m.group(1)]}:{m.group(2)} ', inp)
  # get rid of the prefix phrases
  simplified_query = re.sub(r'PREFIX\s+\w+:\s*<[^>]+>\s*', '', simplified_query)
  # prepend the corresponding subject of each triple contraction to the triple
  for ex in exceptions:
    simplified_query = re.sub(escape_special_characters(rf';\s*{ex[1]}\s+{ex[2]}\s*'), f'. {ex[0]} {ex[1]} {ex[2]} ', simplified_query)
  return simplified_query

def prefixes_update(prefixes, subs):
  for k, v in subs.items():
    if k not in prefixes.keys():
      prefixes[k] = v
  return prefixes

prefix_pattern = re.compile(r'PREFIX\s+([a-z0-9]+):\s*(\S+)', re.IGNORECASE)
where_pattern = re.compile(r'WHERE\s*\{(.+)\}', re.IGNORECASE)
curly_pattern = re.compile(r'\{([^{}]*)\}', re.IGNORECASE)
initial_path = f'../data/LCQUAD1/LCQUAD1_train.json'
final_path = f'../data/LCQUAD1/LCQUAD1_train_final.json'

new_samples = []
prefixes = {}
with open(initial_path, 'r') as f:
  samples = json.load(f)
  for idx in trange(len(samples)):
    sample = samples[idx]
    q = sample['query']
    # print(q)
    answers = {}
    # print(sample)
    answers_length = len(list(sample['answers'].keys()))
    assert answers_length <= 1
    if answers_length == 1:
      answers_values = next(iter(sample['answers'].values()))
      answers_key = next(iter(sample['answers'].keys()))
      answers[answers_key] = []
      if type(answers_values) == list:
        for answer in answers_values:
          answers[answers_key].append(answer)
      elif type(answers_values) == bool:
        answers[answers_key].append(answer)
      else:
        print('error')
    substitutes = {prefix: id for prefix, id in re.findall(prefix_pattern, q)}
    prefixes = prefixes_update(prefixes, substitutes)
    where_triples, curly_triples = [], []
    where_content = re.search(where_pattern, q).group(1)
    if '{' in where_content:
      assert '}' in where_content
      curly_content = re.findall(curly_pattern, q)
      for c in curly_content:
        curly_triples.extend(extract_triples(c))
    if not curly_triples:
      where_triples.extend(extract_triples(where_content))
    # replace each prefix with its corresponding entity/relation id
    triples = where_triples if not curly_triples else curly_triples
    for x in range(len(triples)):
      for y in range(len(triples[x])):
        for z in range(len(triples[x][y])):
          for prefix, sub in substitutes.items():
            if prefix == triples[x][y][z].split(':')[0]:
              triples[x][y][z] = sub + triples[x][y][z][len(prefix):]

    # prepend to each <relation, object> its corresponding subject
    # +
    # get rid of non-triples: 'filter'
    exceptions = []
    for x in range(len(triples)):
      for y in range(len(triples[x])):
        if triples[x][y][0] == 'filter':
          _ = triples.pop(x)
          continue
        if len(triples[x][y]) == 2:
          subject = triples[x][0][0]
          triples[x][y].insert(0, subject)
          exceptions.append(triples[x][y])
    sample['triples'] = triples

    # generate complete triples by replacing ?var with the answers
    new_triples = deepcopy(triples)
    for x in range(len(triples)):
      subs = []
      for y in range(len(triples[x])):
        assert len(answers.keys()) <= 1
        if len(answers.keys()) == 1:
          var = next(iter(answers))
          if ('?' + var) in triples[x][y]:
            i = triples[x][y].index(('?' + var))
            for ans in answers[var]:
              new_triples[x].append(deepcopy(triples[x][y]))
              new_triples[x][-1][i] = ans
            subs.append(y)
          elif ('?' + var + '.') in triples[x][y]:
            i = triples[x][y].index(('?' + var + '.'))
            for ans in answers[var]:
              new_triples[x].append(deepcopy(triples[x][y]))
              new_triples[x][-1][i] = ans
            subs.append(y)
      new_triples[x] = list(filter(lambda r: not (new_triples[x].index(r) in subs), new_triples[x]))

    sample['new_triples'] = new_triples
    new_samples.append(sample)
with open(final_path, 'w') as f:
  json.dump(new_samples, f, indent=4)