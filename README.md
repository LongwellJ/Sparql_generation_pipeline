pipeline for dbpedia datasets:

1.
get new_answers in data from collab script that sends query to api and gets results (redo this in a timely manner)
Mahdiyar_ChatGPT_script.ipynb

2.
get the qald 9 train and test json dataset through preprocess.py or preprocess_original or new_preprocess or new_preprocess.py maybe to get the triples and new triples data column (review this)

3.
get entities from refined model using refined_test_QALD.py 377 408 0.9240196078431373 train and 140 150 0.9333333333333333 test have found at least one entity i think. although that entity may or may not be 'none' it will be "entites": [] in the data if refined didn't find an entity. 24 "None" in train and 10 in test. Not every example with a "None" has no other entities. Updated refine_test_QALD.py to remove the "None" in computation 367 408 0.8995098039215687 train and 135 150 0.9 in test updated

4. noisy_triple_retriever.py gets the triples in a one hop neighbourhood
with the entities extracted, query the dbpedia endpoint to retrieve the one hop neighbourhood around each entity and store them
we query the dbpedia endpoint with noisy_triple_retriever.py

5. make the dataset of triples and their associated NLQ with cross_encoder_data_maker.py. Send each triple with the nlq through the cross encoder after training it on server to perform binary classification. cross_encoder_deberta_xxxx_yyyy.py where xxxx is the mode of the model and yyyy is the dataset name

6. reconnect each of the triples to their questions in the main dataset with combine_triples.py. rank based on cross encoder score and append to dataset. 

7. Construct prompts and train and test model on server with train_Mistral_QALD_retriever.py and Mistral_QALD_trainer_test.py

8. post process with prompt_parser.ipynb and execute queries on the dbpedia endpoint with Mahdiyar_ChatGPT_script.ipynb compare final lists to gold lists and calculate metrics



ELASTIC SEARCH STUFF

query_elastic.py and new_index_dBpedia.py





use create_jsonl_flagembedding_data.py to turn 


torchrun --nproc_per_node 2 -m FlagEmbedding.baai_general_embedding.finetune.run --output_dir home/jlongwel/retriever/similarity --model_name_or_path dunzhang/stella_en_400M_v5 --train_data ./output.jsonl --learning_rate 1e-5 --num_train_epochs 5 --per_device_train_batch_size 1 --dataloader_drop_last True --normlized True --temperature 0.02 --query_max_len 2000 --passage_max_len 2000 --train_group_size 20 --negatives_cross_device --logging_step 10  --query_instruction_for_retrieval "Retrieve the relevant documents given the following context: " --fp16  --save_strategy "epoch" --save_total_limit 4

