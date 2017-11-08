from utils import *
from config import Config


def main():
    conf = Config()
    vocab, idx2char = build_char_vocab(conf.train_path)
    source_seqs, target_seqs = load_data(conf.train_path, vocab, conf.data_path)


if __name__ == "__main__":
    main()
