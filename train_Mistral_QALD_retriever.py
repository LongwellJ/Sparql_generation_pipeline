#from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer, AdamW
from trl import SFTTrainer
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import json, os
import torch
import sys
from tqdm import tqdm
torch.cuda.empty_cache()
def preprocess_text(text):
  # punkt = "!#$%&()*+,./:;<=>?@[\\]^_`{|}~"
  # translator = str.maketrans('', '', punkt)
  # text_without_punctuation = text.translate(translator).lower()
  # processed_string = re.sub(r'\s+', ' ', text_without_punctuation)
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

# train_filepath = f'../data/qald-9-plus/qald_9_plus_train_dbpedia_en_triples_correct_final_with_bm25_triples.json'
# triples_filepath = f'../data/qald-9-plus/cross_encoder_QALD_filtered_toy'
#train_filepath = f'../data/qald-9-plus/qald9_train_with_entites_and_relations.json'
train_filepath = f'../data/qald-9-plus/QALD_Mistral_train_retrieved_triples.json'
#test_filepath = f'../data/qald-9-plus/QALD_Mistral_test_retrieved_triples.json'
with open(train_filepath, 'r') as f: 
  train_data = json.load(f)
  #train_y = list(map(lambda x: '<s> [INST] translate English to SPARQL w.r.t. the following relevant triples:' + ' <sep> ' + preprocess_text(x['question']) + ' <sep> ' + preprocess_text(lists_to_string(x['new_triples'])) + '[/INST]' + preprocess_text(x['new_query']), train_data))
  #train_x = list(map(lambda x: '<s> [INST] translate English to SPARQL w.r.t. the following relevant triples:' + ' <sep> ' + preprocess_text(x['question']) + ' <sep> ' + preprocess_text(lists_to_string(x['new_triples'])) + '[/INST]', train_data))
  # train_x = list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n \'\'\' \n' + preprocess_text(x['question']) + '\n \'\'\' \n Use the following relevant triples to generate the SPARQL query: \n ' + preprocess_text(lists_to_string(x['new_triples']))+'[/INST] \n Output: ', train_data))
  # train_y = list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n ' + preprocess_text(x['question']) + '\n Use the following relevant triples to generate the SPARQL query: \n ' + preprocess_text(lists_to_string(x['new_triples']))+'[/INST] \n Output: The correct SPARQL query to answer the questions is '+ preprocess_text(x['new_query'])+'</s>', train_data))
  #train_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n \'\'\' \n' + preprocess_text(x['question']) + '\n\'\'\' \n'+ '[/INST]' ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['new_query'])+'</s>', train_data))
  #train_y = list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n \'\'\' \n' + preprocess_text(x['question']) + '\n\'\'\' \n'+ '[/INST]' ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['new_query'])+'</s>', train_data))
  #train_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant triples to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(lists_to_string(x['retrieved_triples']))+ '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_y= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant triples to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(lists_to_string(x['retrieved_triples']))+ '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant entities to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['entites']))+'\n\'\'\'\n'+ 'Use the following relevant relations to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['relations'])) + '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_y= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant entities to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['entites']))+'\n\'\'\'\n'+ 'Use the following relevant relations to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(process(x['relations'])) + '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_x= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(process(x['entites']))+', '+ preprocess_text(process(x['relations'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion:'+ preprocess_text(x['question']) +'\nAnswer:[/INST]'+ 'The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_y= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(process(x['entites']))+', '+ preprocess_text(process(x['relations'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion:'+ preprocess_text(x['question']) +'\nAnswer:[/INST]'+ 'The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  train_x= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(lists_to_string(x['retrieved_triples'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion: '+ preprocess_text(x['question']) +'\nAnswer: [/INST]'+ ' The correct SPARQL query to answer the question is '+ preprocess_text(x['query'])+'</s>', train_data))
  train_y= list(map(lambda x: '<s>[INST] Context information is below.\n---------------------\n'+preprocess_text(lists_to_string(x['retrieved_triples'])) +'\n---------------------\nGiven the context information and not prior knowledge, write a query in SPARQL to answer the following question.\nQuestion: '+ preprocess_text(x['question']) +'\nAnswer: [/INST]'+ ' The correct SPARQL query to answer the question is '+ preprocess_text(x['query'])+'</s>', train_data))

  #train_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
  #train_y= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n[/INST]'+ ' The correct SPARQL query to answer the questions is '+ preprocess_text(x['query'])+'</s>', train_data))
#print(train_x[302])
  

# with open(test_filepath, 'r') as f:
#   test_data = json.load(f)
#   val_id=list(map(lambda x: x['id'], test_data))
#   val_new_answers = list(map(lambda x: x['new_answers'], test_data))
#   test_y = list(map(lambda x: preprocess_text(x['query']), test_data))
#   val_question = list(map(lambda x: x['question'], test_data))
#   test_x= list(map(lambda x: '<s>[INST] Write a query in SPARQL to answer the following question: \n\'\'\'\n' + preprocess_text(x['question']) + '\n\'\'\'\n'+ 'Use the following relevant triples to generate the SPARQL query:\n\'\'\'\n' + preprocess_text(lists_to_string(x['retrieved_triples']))+ '\n\'\'\'\n[/INST]', test_data))



class QuestionToSPARQLDataset(Dataset):
    def __init__(self, tokenizer, questions, sparql_queries, max_length=2000):
        self.tokenizer = tokenizer
        self.questions = questions
        self.sparql_queries = sparql_queries
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        sparql_query = self.sparql_queries[idx]

        input_encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            sparql_query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        inputs = {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten(),
        }

        return inputs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

selected_gpu = sys.argv[1]
device = torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
questions = train_x
sparql_queries = train_y
#val_questions = test_x
#val_sparql_queries = test_y
#print(questions[0])
dataset = QuestionToSPARQLDataset(tokenizer, questions, sparql_queries)
#val_dataset = QuestionToSPARQLDataset(tokenizer, val_questions, val_sparql_queries)
#print(type(dataset))
for i in range(407):
   token_count = len(dataset[i]['input_ids'])
   if token_count > 2000:
        print(f"Datapoint {i} exceeds max token length with {token_count} tokens.")

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
    quantization_config=bnb_config, 
    use_cache=False, 
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pretraining_tp = 1
model.config.padding_side = 'right'
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.gradient_checkpointing_enable()
# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    r=1024,
    lora_alpha=2048,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     # Compute your metrics here
#     loss = torch.nn.CrossEntropyLoss()
#     loss_value = loss(logits, labels)
#     return {"eval_loss": loss_value}

model_args = TrainingArguments(
    output_dir="mistral-7b-5-RT",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    #gradient_accumulation_steps=2,# I think this is why even with a batch size of 1, we have half the tqdm as I expect. This also greatly decreases training speed but also decreases GPU costs
    #gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    learning_rate=2e-6,
    save_total_limit=5,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    #lr_scheduler_type="constant",
    disable_tqdm=False
)

# Supervised Fine-Tuning Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #eval_dataset = val_dataset
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    args=model_args
    #compute_metrics=compute_metrics
)

# train
trainer.train()
#save again just in-case
# trainer.save_model()
#model = model.to(device)
#optimizer = bnb.optim.Adam8bit(model.parameters(), lr=6e-6)
#optimizer = AdamW(model.parameters(), lr=6e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)
#num_epochs = 20
#training
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0

#     for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}'):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         sparql_queries = batch['labels'].to(device)
#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=sparql_queries)
#         # print(torch.cuda.memory_summary(device=None, abbreviated=False))
#         # print(torch.cuda.memory_reserved())
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#         del input_ids
#         del attention_mask
#         del sparql_queries

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f'Epoch {epoch + 1} - Average Loss: {avg_loss}')
# model_name = 'mistral_instruct_v2_20_retrieved_triples_20epochs'
# model.save_pretrained(f'../models/{model_name}')
#tokenizer.save_pretrained(f'../models/{model_name}')
# print("Model has been saved.")





