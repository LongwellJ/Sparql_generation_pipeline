import json
import random

def extract_label(uri):
    """Extract the label from a URI."""
    if uri.startswith('<') and uri.endswith('>'):
        uri = uri[1:-1]
    return uri.split('/')[-1].replace('_', ' ')

def process_input_data(input_data, neg_sample_size=19):
    """Process the input data and return a list of output JSON objects."""
    data_by_question = {}

    for item in input_data:
        question = item['question']
        label = item['label']
        triple = item['triple'][:3]  # Take the first 3 elements of triple

        # Convert the triple to a string representation
        subj_label = extract_label(triple[0])
        pred_label = extract_label(triple[1])
        obj_label = extract_label(triple[2])
        triple_str = f"{subj_label}, {pred_label}, {obj_label}"

        # Initialize the dictionary for the question if not already done
        if question not in data_by_question:
            data_by_question[question] = {'pos': [], 'neg': []}

        if label == 1:
            data_by_question[question]['pos'].append(triple_str)
        else:
            data_by_question[question]['neg'].append(triple_str)

    # Prepare the output data
    output_data = []
    for question, lists in data_by_question.items():
        pos_list = lists['pos']
        neg_list = lists['neg']

        # Skip the question if pos_list is empty
        if not pos_list:
            continue  # Skip this entry

        # For each positive example, create a new row with a new random sample of negatives
        for pos_triple in pos_list:
            # Randomly sample negatives for each positive example
            if len(neg_list) > neg_sample_size:
                sampled_neg_list = random.sample(neg_list, neg_sample_size)
            else:
                sampled_neg_list = neg_list

            output_item = {
                'query': question,
                'pos': [pos_triple],
                'neg': sampled_neg_list
            }
            output_data.append(output_item)

    return output_data

# Load input data from a JSON file
with open(f'../data/qald-9-plus/cross_encoder_QALD_filtered.json', 'r') as f:
    input_data = json.load(f)

# Process the input data
output_data = process_input_data(input_data)

# Write the output data to a JSONL file
with open('output.jsonl', 'w') as f:
    for item in output_data:
        json_line = json.dumps(item)
        f.write(json_line + '\n')

# Print the output data for reference (optional)
# for item in output_data:
#     print(json.dumps(item, indent=2))
