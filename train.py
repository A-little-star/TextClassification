import torch
import torch.nn as nn
from torch import optim
from models import Model
from dataset import data_loader, text_ClS
from configs import Config

cfg = Config()

data_path = "./data/nd_train.csv"
dict_path = "./dict"

dataset = text_ClS(dict_path, data_path)
train_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)

for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.devices)
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)
        print("epoch is {}, ite is {}, val is {}".format(epoch, i, loss_val))
        loss_val.backward()

        optimizer.step()

    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), "models/{}.pth".format(epoch))

        