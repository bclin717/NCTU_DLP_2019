#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function, division

import json
import math
import os
import random
import time
from io import open

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import optim

from Lab5.DecoderRNN import DecoderRNN
from Lab5.EncoderRNN import EncoderRNN
from Lab5.wordsDataset import wordsDataset

"""==============================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
================================================================================"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True

def showGraph(df):
    plt.figure(figsize=(10, 6))
    plt.title('Training\nLoss/Score/Weight Curve')

    plt.plot(df.index, df.kl, label='KLD', linewidth=3)
    plt.plot(df.index, df.crossentropy, label='CrossEntropy', linewidth=3)

    plt.xlabel('epoch')
    plt.ylabel('loss')

    h1, l1 = plt.gca().get_legend_handles_labels()

    ax = plt.gca().twinx()
    ax.plot(metrics_df.index, metrics_df.score, '-', label='BLEU4-score', c="C2")
    ax.plot(metrics_df.index, metrics_df.klw, '--', label='KLD_weight', c="C3")
    ax.plot(metrics_df.index, metrics_df.tfr, '--', label='Teacher ratio', c="C4")
    ax.set_ylabel('score / weight')

    h2, l2 = ax.get_legend_handles_labels()

    ax.legend(h1 + h2, l1 + l2)
    plt.show()


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    return sentence_bleu([reference], output, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def __save_model(model_name, model, root):
    if not os.path.isdir(root):
        os.mkdir(root)
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    torch.save(model.state_dict(), p)
    return p


def save_model(models, root='./model'):
    p = {}
    for k, m in models.items():
        p[k] = __save_model(k, m, root)
    return p


def __load_model(model_name, model, root):
    p = os.path.join(root, '{}-params.pkl'.format(model_name))
    if not os.path.isfile(p):
        msg = "No model parameters file for {}!".format(model_name)
        return print(msg)
        raise AttributeError(msg)
    paras = torch.load(p)
    model.load_state_dict(paras)


def load_model(models, root='./model'):
    for k, m in models.items():
        __load_model(k, m, root)


def save_model_by_score(models, bleu_score, root):
    p = os.path.join(root, 'score.json')
    previous = None

    if np.isnan(bleu_score):
        raise AttributeError("BLEU score become {}".format(bleu_score))
        return

    if os.path.isfile(p):
        with open(p, 'r') as f:
            previous = json.load(f)

    if previous is not None and previous['score'] > bleu_score:
        return;

    save_model(models, root)
    previous = {'score': bleu_score}
    with open(p, 'w') as f:
        json.dump(previous, f)

def decode_inference(decoder, z, c, maxlen, teacher=False, inputs=None):
    sos_token = trainDataset.chardict.word2index['SOS']
    eos_token = trainDataset.chardict.word2index['EOS']
    z = z.view(1, 1, -1)

    outputs = []
    x = torch.LongTensor([sos_token]).to(device)
    hidden = decoder.initHidden(z, c)

    i = 0
    for i in range(maxlen):
        x = x.detach()
        output, hidden = decoder(x, hidden)
        outputs.append(output)
        output_onehot = torch.max(torch.softmax(output, dim=1), 1)[1]

        # meet EOS
        if output_onehot.item() == eos_token and not teacher:
            break

        if teacher:
            x = inputs[i + 1:i + 2]
        else:
            x = output_onehot

    # get (seq, word_size)
    if len(outputs) != 0:
        outputs = torch.cat(outputs, dim=0)
    else:
        outputs = torch.FloatTensor([]).view(0, word_size).to(device)

    return outputs


def evaluation(encoder, decoder, dataset, show=True):
    encoder.eval()
    decoder.eval()

    blue_score = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        if dataset.train:
            inputs, input_condition = data
            targets = inputs
            target_condition = input_condition
        else:
            inputs, input_condition, targets, target_condition = data

        # input no sos and eos
        z, _, _ = encoder(inputs[1:-1].to(device), encoder.initHidden(), input_condition)

        # input has sos and eos

        outputs = decode_inference(decoder, z, encoder.condition(target_condition), maxlen=len(targets))

        # show output by string
        outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
        inputs_str = trainDataset.chardict.stringFromLongtensor(inputs, check_end=True)
        targets_str = trainDataset.chardict.stringFromLongtensor(targets, check_end=True)
        outputs_str = trainDataset.chardict.stringFromLongtensor(outputs_onehot, check_end=True)

        if show:
            print(inputs_str, '\nGround Truth :  ', targets_str, ' \nPrediction :', outputs_str, '\n')

        blue_score.append(compute_bleu(outputs_str, targets_str))

    if show:
        print('BLEU-4 score : {}'.format(sum(blue_score) / len(blue_score)))

    return blue_score


def KLD_weight_annealing(epoch):
    w = (epoch % (1.0 / 0.001) * 2) * 0.001
    return min(w, 1.0)


def KL_loss(m, logvar):
    return torch.sum(0.5 * (-logvar + (m ** 2) + torch.exp(logvar) - 1))


def trainIters(name, encoder, decoder, epoch_size, learning_rate=1e-2, show_size=1000, KLD_weight=0.0,
               teacher_forcing_ratio=0.5, eval_size=100, metrics=[], start_epoch=0):
    start = time.time()
    show_loss_total = 0
    plot_loss_total = 0
    plot_kl_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(start_epoch, epoch_size):
        encoder.train()
        decoder.train()

        # get data from trian dataset
        for idx in range(len(trainDataset)):
            data = trainDataset[idx]
            inputs, c = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # input no sos and eos
            z, m, logvar = encoder(inputs[1:-1].to(device), encoder.initHidden(), c)

            # decide teacher forcing
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            # input has sos
            outputs = decode_inference(
                decoder, z, encoder.condition(c), maxlen=inputs[1:].size(0),
                teacher=use_teacher_forcing, inputs=inputs.to(device))

            # target no sos
            output_length = outputs.size(0)

            loss = criterion(outputs, inputs[1:1 + output_length].to(device))
            kld_loss = KL_loss(m, logvar)

            (loss + (KLD_weight(epoch) * kld_loss)).backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            show_loss_total += loss.item() + (KLD_weight(epoch) * kld_loss.item())
            plot_loss_total += loss.item()
            plot_kl_loss_total += kld_loss.item()

            # show output by string
            outputs_onehot = torch.max(torch.softmax(outputs, dim=1), 1)[1]
            inputs_str = trainDataset.chardict.stringFromLongtensor(inputs, show_token=True)
            outputs_str = trainDataset.chardict.stringFromLongtensor(outputs_onehot, show_token=True)

            if np.isnan(loss.item()) or np.isnan(kld_loss.item()):
                raise AttributeError("Became NAN !! loss : {}, kl : {}".format(loss.item(), kld_loss.item()))

        score = 0
        for _ in range(eval_size):
            all_score = evaluation(encoder, decoder, testDataset, show=False)
            score += sum(all_score) / len(all_score)
        score /= eval_size

        save_model_by_score(
            {'encoder': encoder, 'decoder': decoder},
            score,
            os.path.join('.', name)
        )

        if (epoch + 1) % show_size == 0:
            show_loss_total /= show_size
            print("{} ({} {}%) \ntotal loss : {:.4f}".format(
                timeSince(start, (epoch + 1) / epoch_size),
                epoch + 1, (epoch + 1) * 100 / epoch_size, show_loss_total
            ))
            print('bleu score : {:.5f}\n'.format(score))
            show_loss_total = 0

        metrics.append((
            plot_loss_total, plot_kl_loss_total, score,
            KLD_weight(epoch), teacher_forcing_ratio, learning_rate
        ))

        plot_loss_total = 0
        plot_kl_loss_total = 0

    return metrics





# use gaussian noise to generate test data
def generate_word(encoder, decoder, z, condition, maxlen=20):
    encoder.eval()
    decoder.eval()

    outputs = decode_inference(
        decoder, z, encoder.condition(condition), maxlen=maxlen
    )

    return torch.max(torch.softmax(outputs, dim=1), 1)[1]


def generate_test(encoder, decoder, noise=None):
    if noise is None:
        noise = encoder.sample_z()

    strs = []
    for i in range(len(trainDataset.tenses)):
        outputs = generate_word(encoder, decoder, noise, i)
        output_str = trainDataset.chardict.stringFromLongtensor(outputs)
        print('{:20s} : {}'.format(trainDataset.tenses[i], output_str))
        strs.append(output_str)

    print("")
    return noise, strs


torch.max(torch.softmax(torch.randn(1, 28), dim=1), 1)[1]
trainDataset = wordsDataset()
testDataset = wordsDataset(False)
word_size = trainDataset.chardict.n_words
num_condition = len(trainDataset.tenses)
# ----------Hyper Parameters----------#
hidden_size = 256
latent_size = 32
condition_size = 8
teacher_forcing_ratio = 0.5
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05


def main():
    metrics = []
    encoder = EncoderRNN(
        word_size, hidden_size, latent_size, num_condition, condition_size
    ).to(device)
    decoder = DecoderRNN(
        word_size, hidden_size, latent_size, condition_size
    ).to(device)

    # load_model(
    #     {'encoder': encoder, 'decoder': decoder},
    #     os.path.join('.', 'best')
    # )

    trainIters('training_from_init', encoder, decoder, epoch_size=250, show_size=5, learning_rate=10e-4,
               KLD_weight=KLD_weight_annealing, teacher_forcing_ratio=teacher_forcing_ratio, metrics=metrics,
               start_epoch=len(metrics))

    torch.save(metrics, os.path.join('.', 'metrics.pkl'))
    metrics_df = pd.DataFrame(metrics, columns=[
        "crossentropy", "kl", "score", "klw", "tfr", "lr"
    ])
    metrics_df.head()

    showGraph(metrics_df)

    all_score = evaluation(encoder, decoder, testDataset)

    for i in range(0, 4):
        noise = encoder.sample_z()
        generate_test(encoder, decoder, noise)


if __name__ == '__main__':
    main()
