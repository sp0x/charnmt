import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os
import re
import random

from config import Config
import utils
from CharNMT import CharNMT
from generate import generator


def loss_in_batch(output, label, mask, loss_fn):
    loss = 0
    for i in range(len(output)):
        loss += loss_fn(output[i:i+1], label[i:i+1]) * mask[i]
    return loss


def train(model, source, target, lr, conf, idx2char):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    total_loss = 0
    data_size = len(source)

    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size, False)):
        batch_size, max_len = x.shape
        x = Variable(torch.Tensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = model.encoder.init_gru(batch_size)

        model.zero_grads()
        batch_loss = 0

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        context, dec_h = model.encoder(x, enc_h)

        translation = []
        gen = generator(model, idx2char)
        for i in range(1, y.size(1)):
            next_char, dec_h, attn = model(y[:,i-1], context, dec_h)
            batch_loss += loss_in_batch(next_char, y[:,i], mask[:,i], loss_fn)
            char_idx = next_char.data.topk(1)[1][0][0]
            translation.append(char_idx)
        batch_loss /= batch_size
        print(translation)

        batch_loss.backward()
        opt.step()

        total_loss += batch_loss.data[0] * batch_size

        if (batch + 1) % conf.log_interval == 0:
            size = conf.batch_size * batch + batch_size
            print("[{:5d}/{:5d}] batches\tLoss: {:5.6f}"
                    .format(size, data_size, total_loss / size))

    return total_loss / data_size


def evaluate(model, source, target, conf):
    model.eval()
    loss_fn = nn.NLLLoss()

    total_loss = 0
    data_size = len(source)

    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size)):
        batch_size, max_len = x.shape
        x = Variable(torch.Tensor(x.tolist()), volatile=True)
        y = Variable(torch.LongTensor(y.tolist()))

        enc_h = model.encoder.init_gru(batch_size)

        if conf.cuda:
            x = x.cuda()
            y = y.cuda()
            enc_h = enc_h.cuda()

        batch_loss = 0

        context, dec_h = model.encoder(x, enc_h)

        for i in range(1, y.size(1)):
            next_char, dec_h, attn = model(y[:,i-1], context, dec_h)
            batch_loss += loss_in_batch(next_char, y[:,i], mask[:,i], loss_fn)

        total_loss += batch_loss.data[0]

    return total_loss / data_size


def lr_schedule(t):
    """
    learning rate schedule, use piecewise
    """
    if t < 10:
        return 0.0005

    if t < 50:
        return 0.0002

    return 0.0001


def get_latest_saver(path, prefix):
    """
    return the latest saved model name and epoch number
    """
    files = os.listdir(path)

    latest = -1
    file_name = ""
    pattern = re.compile(r"{}_(\d+)".format(prefix))
    for f in files:
        if f.startswith(prefix):
            epoch = int(re.findall(pattern, f)[0])
            if latest < epoch:
                latest = epoch
                file_name = f

    return os.path.join(path, file_name), latest


def main():
    # Load data
    conf = Config()
    vocab, idx2char = utils.build_char_vocab(
            [conf.train_path, conf.dev_path, conf.test_path])
    print("vocabulary size = {}".format(len(vocab)))

    train_source_seqs, train_target_seqs = utils.load_data(
            conf.train_path, 
            vocab, 
            conf.train_pickle,
            conf.max_seq_len, 
            conf.reverse_source)
    print("{} source and {} target sequences of training set loaded.".format(
        len(train_source_seqs), len(train_target_seqs)))
    """
    dev_source_seqs, dev_target_seqs = utils.load_data(
            conf.dev_path, 
            vocab, 
            conf.dev_pickle,
            conf.max_seq_len, 
            conf.reverse_source)
    print("{} source and {} target sequences of validation set loaded.".format(
        len(dev_source_seqs), len(dev_target_seqs)))
    test_source_seqs, test_target_seqs = utils.load_data(
            conf.test_path, 
            vocab, 
            conf.test_pickle,
            conf.max_seq_len, 
            conf.reverse_source)
    print("{} source and {} target sequences of test set loaded.".format(
        len(test_source_seqs), len(test_target_seqs)))
    """

    # Define model
    model = CharNMT(len(vocab), 
            conf.max_seq_len, 
            conf.source_emb, 
            conf.target_emb, 
            conf.hid_dim, 
            conf.dropout, 
            conf.stride)

    if conf.cuda:
        model.cuda()

    min_loss = float("inf")
    lr = conf.lr

    # Check latest saved model
    save_file, start_epoch = get_latest_saver(conf.save_path, model.name)

    if conf.debug_mode:
        debug_size = int(conf.batch_size * 1)
        train_source_seqs = train_source_seqs[:debug_size]
        train_target_seqs = train_target_seqs[:debug_size]
    else:
        size = 10000
        idx = (start_epoch // conf.epochs) % (len(train_source_seqs) // size)
        idx *= size
        train_source_seqs = train_source_seqs[idx:idx+size]
        train_target_seqs = train_target_seqs[idx:idx+size]

    if start_epoch >= 0 and not conf.debug_mode:
        try:
            model = torch.load(save_file)
            min_loss = evaluate(model, dev_source_seqs, dev_target_seqs, conf)
            print("Initial validation loss: {:5.6f}".format(min_loss))
        except RuntimeError as e:
            print("[loading existing model error] {}".format(str(e)))
    else:
        start_epoch = 0

    # Training
    for epoch in range(start_epoch, conf.epochs+start_epoch):
        lr = lr_schedule(epoch)
        print("*** Epoch [{:5d}] lr = {} ***".format(epoch, lr))

        train_loss = train(model, train_source_seqs, train_target_seqs, lr, conf, idx2char)
        print("Training set\tLoss: {:5.6f}".format(train_loss))

        if conf.debug_mode:
            continue

        dev_loss = evaluate(model, dev_source_seqs, dev_target_seqs, conf)
        print("Validation set\tLoss: {:5.6f}".format(dev_loss))

    save_file = os.path.join(conf.save_path, 
            "{}_{}".format(model.name, conf.epochs+start_epoch))
    with open(save_file, "wb") as f:
        torch.save(model, f)

    # Evaluate test set
    if not conf.debug_mode:
        test_loss = evaluate(model, test_source_seqs, test_target_seqs, conf)
        print("Test set loss after {:4d} epochs: {:5.6f}".format(
            conf.epochs+start_epoch, test_loss))


if __name__ == "__main__":
    main()
