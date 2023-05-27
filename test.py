import torch
import torch.nn as nn
from torch import optim
from models import Model
from dataset import data_loader, text_ClS
from configs import Config

cfg = Config()

data_path = "./data/nd_test.csv"
dict_path = "./dict"

dataset = text_ClS(dict_path, data_path)
test_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)
model_text_cls.load_state_dict(torch.load("models/10.pth"))

for i, batch in enumerate(test_dataloader):
    label, data = batch
    data = torch.tensor(data).to(cfg.devices)
    label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)

    pred_softmax = model_text_cls.forward(data)
    print(pred_softmax)
    print(label)

    pred = torch.argmax(pred_softmax, dim=1)
    print(pred)