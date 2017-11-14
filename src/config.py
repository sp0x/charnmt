import os


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
    hid_dim = 300
    stride = 5
    dropout = 0.5
    
    epochs = 5
    batch_size = 32
    lr = 0.0001
    log_interval = 100
    reverse_source = False

    debug_mode = False
    cuda = True
