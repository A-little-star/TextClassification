import torch
from transformers import BertModel, BertTokenizer, BertConfig
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
config = BertConfig.from_pretrained('bert-base-chinese')
config.update({'output_hidden_states':True})
model = BertModel.from_pretrained("bert-base-chinese", config=config)

print(tokenizer.encode("生活的真谛是美和爱"))
print(tokenizer.encode_plus("生活的真谛是美和爱","说的太好了"))

import json
data_gpt = []
a = 0
with open('baike.jsonl', 'r', encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # print(data)
        print(data['question'])
        data_gpt.append(data['chatgpt_answers'][0])
        a += 1
        if a == 3:
            break

# print(data_gpt)
test1 = tokenizer(data_gpt)
# print(test1)
input_ids = test1['input_ids']
token_type_ids = test1['token_type_ids']
print(input_ids)
print(token_type_ids)

input_ids = torch.tensor([input_ids])
token_type_ids = torch.tensor([token_type_ids])
print(input_ids)
print(token_type_ids)
model.eval()

device = 'cpu'
tokens_tensor = input_ids.to(device)
segments_tensors = token_type_ids.to(device)
model.to(device)

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs

# print(encoded_layers)