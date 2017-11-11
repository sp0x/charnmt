import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os

from config import Config
import utils
from CharNMT import CharNMT


def loss_in_batch(output, label, mask, loss_fn):
    loss = 0
    for i in range(len(output)):
        loss += loss_fn(output[i:i+1], label[i:i+1]) * mask[i]
    return loss


def train(model, source, target, lr, conf):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    data_size = len(source)

    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size, True)):
        batch_size, max_len = x.shape
        x = Variable(torch.Tensor(x.tolist()), volatile=False)
        y = Variable(torch.LongTensor(y.tolist()))
        enc_h, dec_h = model.init_hidden(batch_size)

        model.zero_grads()
        batch_loss = 0

        context = model.compute_context(x, enc_h)
        for i in range(1, y.size(1)):
            next_char, dec_h = model(y[:,i-1], context, dec_h)
            batch_loss += loss_in_batch(next_char, y[:,i], mask[:,i], loss_fn)
        batch_loss /= batch_size

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
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    data_size = len(source)

    for batch, (x, y, mask) in enumerate(utils.batchify(
        source, target, conf.stride, conf.batch_size)):
        batch_size, max_len = x.shape
        x = Variable(torch.Tensor(x.tolist()), volatile=True)
        y = Variable(torch.LongTensor(y.tolist()))
        enc_h, dec_h = model.init_hidden(batch_size)

        batch_loss = 0

        context = model.compute_context(x, enc_h)
        for i in range(1, y.size(1)):
            next_char, dec_h = model(y[:,i-1], context, dec_h)
            batch_loss += loss_in_batch(next_char, y[:,i], mask[:,i], loss_fn)

        total_loss += batch_loss.data[0]

    return total_loss / data_size


def main():
    # Load data
    conf = Config()
    vocab, idx2char = utils.build_char_vocab(conf.train_path)
    train_source_seqs, train_target_seqs = utils.load_data(
            conf.train_path, 
            vocab, 
            conf.data_path,
            conf.max_seq_len, 
            conf.reverse_source)

    if conf.debug_mode:
        debug_size = int(conf.batch_size * 1.5)
        train_source_seqs = train_source_seqs[:debug_size]
        train_target_seqs = train_target_seqs[:debug_size]

    # Define model
    model = CharNMT(len(vocab), 
            conf.max_seq_len, 
            conf.source_emb, 
            conf.target_emb, 
            conf.hid_dim, 
            conf.dropout, 
            conf.stride)

    min_loss = float("inf")
    lr = conf.lr

    # Check existent saved model
    save_file = os.path.join(conf.save_path, model.name)
    if os.path.exists(save_file) and not conf.debug_mode:
        try:
            model = torch.load(save_file)
            min_loss = model.evaluate(dev_source_seqs, dev_target_seqs)
            print("Initial validation loss: {:5.6f}".format(min_loss))
        except RuntimeError as e:
            print("[loading existing model error] {}".format(str(e)))

    # Training
    for epoch in range(conf.epochs):
        print("*** Epoch [{:5d}] ***".format(epoch))
        train_loss = train(model, train_source_seqs, train_target_seqs, lr, conf)
        print("Training set\tLoss: {:5.6f}".format(train_loss))

        if conf.debug_mode:
            continue

        dev_loss = evaluate(model, dev_source_seqs, dev_target_seqs, conf)

        if dev_loss < min_loss:
            min_loss = dev_loss
            with open(save_file, "wb") as f:
                torch.save(model, f)

        print("Validation set\tLoss: {:5.6f}".format(dev_loss))


if __name__ == "__main__":
    main()
