from SPARQLWrapper import SPARQLWrapper, JSON
import json
import random

def get_random_elements(input_list, n):
    if n > len(input_list):
        return input_list  # Return the entire list if it's shorter than n
    return random.sample(input_list, n)

def get_dbpedia_triples(entity_uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = """
    SELECT ?subject ?predicate ?object
    WHERE {
        { dbr:%s ?predicate ?object . FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) .
        FILTER NOT EXISTS {
            FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
        }}
        UNION
        { ?subject ?predicate dbr:%s . FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) .
        FILTER NOT EXISTS {
            FILTER (REGEX(STR(?predicate), "wikiPageLink$") || REGEX(STR(?predicate), "wikiPageUsesTemplate$") || REGEX(STR(?predicate), "rdf-schema#seeAlso$") || REGEX(STR(?predicate), "wikiPageWikiLink$") || REGEX(STR(?predicate), "wikiPageRedirects$") || REGEX(STR(?predicate), "wikiPageDisambiguates$"))
        }}
    } LIMIT 20
    """ % (entity_uri, entity_uri)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    triples = []
    for result in results["results"]["bindings"]:
        subject = f'<{result["subject"]["value"]}>' if "subject" in result else f'dbr:{entity_uri}'
        predicate = f'<{result["predicate"]["value"]}>' if "predicate" in result else f'dbr:{entity_uri}'
        obj = f'<{result["object"]["value"]}>' if "object" in result else f'dbr:{entity_uri}'
        triples.append([subject, predicate, obj])

    return triples


# Example usage
with open("LCQUAD1_train_final_with_bm25_triples.json", "r", encoding='utf-8') as file:
    test_data = json.load(file)
    answers = list(map(lambda x: x['query'], test_data))
    question = list(map(lambda x: x['question'], test_data))
    ans = list(map(lambda x: x['answers'], test_data))
    #new_answers = list(map(lambda x: x['new_answers'], test_data))
    triples = list(map(lambda x: x['triples'], test_data))
    new_triples = list(map(lambda x: x['new_triples'], test_data))
    bm25_triples = list(map(lambda x: x['bm25_triples'], test_data))
    ids = list(map(lambda x: x['id'], test_data))

results = []
count=0
for uri in new_triples:
    string_list = []
    noisy_triples = []
    extended_counter = 0
    k=len(uri)
    #print(k)
    for j in range(k):
        #print(j)
        #print(uri[j])
        for i in uri[j]:
            #print(i)
            for l in i:
                #print(l)
                entity_name = l.split("/")[-1]
                entity_name = entity_name.replace(">","")
                
                if entity_name in string_list:
                    print#("This entity has already been processed", entity_name)
                    continue
                else:
                    string_list.append(entity_name)
                    #print(string_list)
                    print#("This entity has not been processed", entity_name)
                    try:
                        triples = get_dbpedia_triples(entity_name)
                        noisy_triples.extend(triples)
                        extended_counter+=1
                        #print("The number of triples retrieved is ", extended_counter)
                    except:
                        #print("Error, that entity did not return anything")
                        continue
            if extended_counter >= 5:
                #print("The number of triples retrieved is greater than 5 ")
                break
        if extended_counter >= 5:
            #print("The number of triples retrieved is greater than 5 ")
            break      


    results.append({'id': ids[count], 'question': question[count],'answers':ans[count],'new_triples': new_triples[count], 'bm25_triples': bm25_triples[count], 'noisy_triples': noisy_triples})
    count+=1
    print("the count is ", count)


with open("LCQUAD1_train_final_with_bm25_triples_with_noisy_triples.json", "w", encoding='utf-8') as file:
    json.dump(results, file, indent=4)