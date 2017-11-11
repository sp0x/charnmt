import os


class Config(object):

    data_path = "../data"
    train_path = data_path + "/train.txt"
    save_path = "saver"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    max_seq_len = 450
    source_emb = 128
    target_emb = 512
    hid_dim = 300
    stride = 5
    dropout = 0.5
    
    epochs = 100
    batch_size = 64
    lr = 0.0001
    log_interval = 100
    reverse_source = False

    debug_mode = True
