import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config import Config
import utils

def train(model, source, target):
    model.train()
    opt = optim.Adam()
    total_loss = 0
    data_size = len(source)
    
    for epoch, (x, y) in enumerate(utils.batchify(
            source, target, conf.stride, conf.batch_size, True)):
        batch_size = len(x)
        x = Variable(torch.Tensor(x), volatile=False)
        y = Variable(torch.LongTensor(y))
        
        self.zero_grad()
        


def main():
    conf = Config()
    vocab, idx2char = build_char_vocab(conf.train_path)
    source_seqs, target_seqs = load_data(conf.train_path, 
                                         vocab, 
                                         conf.data_path,
                                         conf.max_seq_len)


if __name__ == "__main__":
    main()
