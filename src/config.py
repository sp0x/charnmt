class Config(object):

    data_path = "../data"
    train_path = data_path + "/train.txt"

    max_seq_len = 450
    source_emb = 128
    target_emb = 512
    hid_dim = 300
    stride = 5
    dropout = 0.5
    
    batch_size = 32
