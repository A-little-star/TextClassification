import torch
class Config():
    def __init__(self):
        self.n_vocab = 1002
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.8
        self.num_classes = 2
        self.pad_size = 3
        self.batch_size = 128
        self.is_shuffle = True
        self.learn_rate = 0.001
        self.num_epochs = 100
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
