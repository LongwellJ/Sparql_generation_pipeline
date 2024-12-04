from SPARQLWrapper import SPARQLWrapper, JSON
import json
import random
from tqdm import tqdm
from tqdm import trange
#def get_random_elements(input_list, n):
    # if n > len(input_list):
    #     raise ValueError("n cannot be larger than the length of the input list")
    # return random.sample(input_list, n)

# def beautify(obj):
#     beautified = json.dumps(obj, indent=4)
#     return beautified

def get_dbpedia_triples(entity_uri, relation_uri=None, type=2):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    if type == 1:

        query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            { %s %s ?object . FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
            UNION
            { ?subject %s %s . FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
        }
        
        """ % (entity_uri, relation_uri,relation_uri, entity_uri)

    elif type == 2:
        query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            { <http://dbpedia.org/resource/%s> ?predicate ?object . FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
            UNION
            { ?subject ?predicate <http://dbpedia.org/resource/%s> . FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
        }
        
        """ % (f'{entity_uri}', f'{entity_uri}')
    elif type == 3:
        query = """
        SELECT ?subject ?predicate ?object
        WHERE {
            { ?subject %s ?object . FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
            UNION
            { ?subject %s ?object . FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) .
            FILTER NOT EXISTS {
                FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
            }}
        }
        
        """ % (relation_uri, relation_uri)

    # query = """
    # SELECT ?subject ?predicate ?object
    # WHERE {
    #     { <http://dbpedia.org/resource/%s> ?predicate ?object . }
    #     UNION
    #     { ?subject ?predicate <http://dbpedia.org/resource/%s> . }
    # }
    
    # """ % (entity_uri, entity_uri)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    #print(beautify(results))

    triples = []
    for result in results["results"]["bindings"]:
        #print(result)
        #subject = f'<{result["subject"]["value"]}>' if "subject" in result.keys() else f'<http://dbpedia.org/resource/{entity_uri}>'
        subject = f'<{result["subject"]["value"]}>' if "subject" in result.keys() else f'{entity_uri}'
        predicate = f'<{result["predicate"]["value"]}>' if "predicate" in result.keys() else f'{relation_uri}'
        #obj = f'<{result["object"]["value"]}>' if "object" in result.keys() else f'<http://dbpedia.org/resource/{entity_uri}>'
        obj = f'<{result["object"]["value"]}>' if "object" in result.keys() else f'{entity_uri}'
        triples.append((subject, predicate, obj))
        #print(triples)
    #print(predicate)
    return triples

# Example usage
entity_uri = f"Alpine_Brigade_Taurinens"  # Replace with the entity you are interested in
print(entity_uri)
#relation_uri = "<http://dbpedia.org/ontology/timeZone>" # Replace with the relation you are interested in
#print(relation_uri)
triples = get_dbpedia_triples(entity_uri)

# # #print('\n'.join(list(map(str, get_random_elements(triples, 50)))))
# # i = 0
# # for triple in triples:
# #     print(triple)
# #     i+=1

# # print(i)
datafilepath = 'LCQUAD1_train_final.json'
with open(datafilepath, 'r') as file:
    data = json.load(file)
    ids = list(map(lambda x: x['id'], data))
    query = list(map(lambda x: x['query'], data))
    question = list(map(lambda x: x['question'], data))
    answers = list(map(lambda x: x['answers'], data))
    #new_answers = list(map(lambda x: x['new_answers'], data))
    triples = list(map(lambda x: x['triples'], data))
    new_triples = list(map(lambda x: x['new_triples'], data))
    entities = list(map(lambda x: x['entites'], data))

results = []
for j in trange(len(ids)):
    retrieved_triples = []
    #print(entities[j])
    for k in entities[j]:
        print(k[0])
        #print(k)
        #print(k[0])
        # for i in relations[j]:
        if k != '':
            k[0]=k[0].replace("\"", "")
            trips = get_dbpedia_triples(k[0])
            retrieved_triples.append(trips)
        #print(retrieved_triples)
    results.append({'id': ids[j], 'query': query[j], 'question': question[j], 'answers': answers[j], 'triples': triples[j], 'new_triples': new_triples[j], 'entities': entities[j], 'retreived_triples': retrieved_triples})
            #print(retrieved_triples)

with open('LCQUAD1_train_final'+'with_retrieved_triples.json', 'w') as file:
    json.dump(results, file, indent=4)










