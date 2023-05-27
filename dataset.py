from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import pandas as pd

def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    a = 0
    for item in dict_list:
        a += 1
        if a == 1:
            continue
        item = item.split(",")
        # print(int(item[1].strip()))
        voc_dict[item[0]] = int(item[1].strip())
    # print(voc_dict)
    return voc_dict

def load_data(data_path):
    data_list = pd.read_csv(data_path)
    # print(data_list)
    data = []
    max_len_seq = 0
    for idx, item in data_list.iterrows():
        label = item['label']
        content = str(item['cutword']).split()
        if (len(content) > max_len_seq):
            max_len_seq = len(content)
        data.append([label, content])
    return data, max_len_seq

data, max_len_seq = load_data("./data/nd_test.csv")
# print(data)

class text_ClS(Dataset):
    def __init__(self, voc_dict_path, data_path):
        self.voc_dict = read_dict(voc_dict_path)
        self.data_path = data_path
        self.data, self.max_len_seq = load_data(self.data_path)
        
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict['<UNK>'])
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict['<PAD>'] for _ in range(self.max_len_seq - len(input_idx))]

        data = np.array(input_idx)
        return label, data


def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)

if __name__ == "__main__":
    data_path = "./data/nd_train.csv"
    dict_path = "./dict"
    train_dataloader = data_loader(dict_path, data_path)
    for i, batch in enumerate(train_dataloader):
        print(batch)