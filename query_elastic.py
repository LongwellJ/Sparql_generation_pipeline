from elasticsearch import Elasticsearch
import json

# Initialize the Elasticsearch client
es = Elasticsearch(["http://localhost:9200"])

# Specify the name of your index
index_name = "dbpedialinks"  # Adjust this to your actual index name

# Path to the JSON file with questions
json_file_path = 'LCQUAD1_train_final.json'

# Function to perform BM25 search for each question
def search_questions(input_path):
    # Load the JSON data
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Iterate over the questions
    for item in data:
        question_text = item.get('question', '')

        # Define the Elasticsearch query
        query = {
            "query": {
                "match": {
                    "processed_triple" : question_text
                }
            },
            "size": 20 # Number of hits to return
        }

        # Execute the search query
        results = es.search(index=index_name, body=query)
        item['bm25_triples'] = [[]]
        for hit in results['hits']['hits']:
            item['bm25_triples'][0].append([hit['_source']['raw_triple']])
        # Optionally, process the results here (e.g., extract and store specific fields)
        # For demonstration, let's just print the number of hits and the first hit's raw_triple
        print(f"Question: {question_text}")
        print(f"Total hits: {results['hits']['total']['value']}")
        if results['hits']['hits']:
            first_hit = results['hits']['hits'][0]
            print(f"Example matching procsessed triple: {first_hit['_source'].get('processed_triple')}")
            print(f"Example matching triple: {first_hit['_source'].get('raw_triple')}")
            
        print("\n----------\n")

    # Save the updated JSON data
        
    # Save the modified data back to a new JSON file
    output_path = input_path.replace('.json', '_with_bm25_triples.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Updated JSON data has been saved to {output_path}")


# Run the function
search_questions(json_file_path)
