import bz2, os, re
import urllib.request
from elasticsearch import helpers
from elasticsearch import Elasticsearch


# Download of data http://downloads.dbpedia.org/2016-04/core-i18n/en/page_links_en.ttl.bz2
filename = "mappingbased-objects_lang=en.ttl.bz2"
indexName = "dbpedialinks"


# Update regex to capture the relation
linePattern = re.compile(r'<http://dbpedia.org/resource/([^>]*)> <http://dbpedia.org/ontology/([^>]*)> <http://dbpedia.org/resource/([^>]*)>.*',
                         re.MULTILINE | re.DOTALL)

es = Elasticsearch(hosts=["http://localhost:9200"])
print("Wiping any existing index...")
# Assuming es is your Elasticsearch client instance and indexName is your index
es.indices.delete(index=indexName, ignore=[400, 404])

# Updated index settings to include processed_triple
indexSettings = {
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
      "properties": {
        "raw_triple": {
          "type": "text"
        },
        "processed_triple": { # New field for the processed triple
          "type": "text"
        }
      }
  }
}


es.indices.create(index=indexName, body=indexSettings)

actions = []
rowNum = 0
lastSubject = ""
article = {}
numLinks = 0
numOrigLinks = 0

def newArticle():
    article = {'raw_triple': "", 'processed_triple': ""}
    return article


with bz2.open(filename, 'rt', encoding='utf-8') as file:
    for line in file:
        m = linePattern.match(line)

        if m:
            subject = urllib.parse.unquote(m.group(1)).replace('_', ' ')
            relation = urllib.parse.unquote(m.group(2)).replace('_', ' ') # Capture the relation
            linkedSubject = urllib.parse.unquote(m.group(3)).replace('_', ' ')

            processed_triple = f"{subject} {relation} {linkedSubject}" # Format the processed triple
            stripped = line.strip()
            stripped = stripped.replace(".","")
            
            article["raw_triple"] = stripped
            article["processed_triple"] = processed_triple
            action = {
                            "_index": indexName,
                            '_op_type': 'index',
                            "_source": article
                        }
                        
            actions.append(action)

            if len(actions) >= 5000:
                helpers.bulk(es, actions)
                actions.clear()


            rowNum += 1

            if rowNum % 100000 == 0:
                print(rowNum, subject, linkedSubject,stripped,processed_triple)

            article = newArticle()
        


# Make sure to process any remaining actions after exiting the loop
if actions:
    helpers.bulk(es, actions)
    actions.clear()