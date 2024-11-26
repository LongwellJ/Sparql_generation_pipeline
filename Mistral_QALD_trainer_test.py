from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import sys
import json
from tqdm import trange
import os
import bitsandbytes as bnb
# fine-tuned model id
model_id = "./mistral-7b-5-RT/checkpoint-1632"

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# load base LLM model, LoRA params and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16

)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pretraining_tp = 1
model.config.padding_side = 'right'
#set cuda gpu
selected_gpu = sys.argv[1]
device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")

#preprocess functions
def preprocess_text(text):
  return text

def lists_to_string(input_lists, limit=5):
  # depth=3
  output_string = ''
  i = 0
  for x in range(len(input_lists)):
    #print(x)
    #print(input_lists[x])
    #for y in range(len(input_lists[x])):
    for y in input_lists[x]:
      #print(y)
      #print(input_lists[x][y])
      #print(y)
      #input_lists[x][y]
      output_string += y+ ' '
      #output_string += ' '.join(input_lists[x][y]) + ' <sep> '
      #print(output_string)
    output_string += '<sep> '
    i += 1
    if i >= limit:
        return output_string.strip()
  return output_string.strip()

def process(input_lists, limit=10):
    output_string = ''
    i=0
    for x in input_lists:
        #print(type(x))
        output_string+=str(x)+' '
        output_string += '<sep> ' 
    i += 1
    if i >= limit:
        return output_string.strip()
    return output_string.strip()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
model = model.to(device)


test_filepath = f'../data/qald-9-plus/QALD_Mistral_test_retrieved_triples.json'
test_results_filepath = f'./results/mistral-7b-5-RT-5_epochs_test.json'
# test_results_filepath = f'./results/t5-3b-tri-10-t_qald-9-plus_v2.json'

with open(test_filepath, 'r') as f:
  test_data = json.load(f)
  id=list(map(lambda x: x['id'], test_data))
  new_answers = list(map(lambda x: x['new_answers'], test_data))
  test_y = list(map(lambda x: preprocess_text(x['query']), test_data))
  question = list(map(lambda x: x['question'], test_data))
  #test_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant entities to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['entites']))+'\n\'\'\'\n'+ 'Use the following relevant relations to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['relations'])) + '\n\'\'\'\n[/INST]', test_data))  
  #test_x= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(process(x['entites']))+', '+ preprocess_text(process(x['relations'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion:'+ preprocess_text(x['question']) +'\nAnswer:[/INST]', test_data))
  test_x= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(lists_to_string(x['retrieved_triples'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion: '+ preprocess_text(x['question']) +'\nAnswer: [/INST]', test_data))


results = []
queries = list(zip(test_x, test_y))
for i in trange(len(queries)):
    test_case, gold = queries[i]
    input_ids = tokenizer.encode(test_case, return_tensors='pt', max_length=2000, truncation=True)
    if len(input_ids) >1024:
      print(f"Datapoint {i} exceeds max token length with {input_ids} tokens.")
    input_ids = input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_length=2000)#not sure if passing attention mask does anything
    generated_query = tokenizer.decode(output[0], skip_special_tokens=True)
    results.append({'id': id[i], 'question': question[i], 'gold_query': gold, 'predicted_query': generated_query, 'new_answers': new_answers[i], 'formulated_prompt': test_case})
    # print(f"Gold SPARQL Query: {gold}")
    # print(f"Generated SPARQL Query: {generated_query}")
    # print()

with open(test_results_filepath, 'w') as f:
  json.dump(results, f, indent=4)