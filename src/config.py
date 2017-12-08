import os
import torch


class Config(object):

    data_path = "../data"
    train_path = data_path + "/train.txt"
    dev_path = data_path + "/dev.txt"
    test_path = data_path + "/test.txt"
    
    train_pickle = data_path + "/train_{}.p"
    dev_pickle = data_path + "/dev_{}.p"
    test_pickle = data_path + "/test_{}.p"

    save_path = "saver"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    max_seq_len = 450
    source_emb = 128
    target_emb = 512
    n_gram_size = 1001
    hid_dim = 300
    stride = 1
    dropout = 0.5
    
    epochs = 100
    batch_size = 1
    lr = 0.001
    log_interval = 50
    reverse_source = False
    word_level = True
    teaching_ratio = 0.5

    bleu_n = 4

    debug_mode = True
    cuda = torch.cuda.is_available()

