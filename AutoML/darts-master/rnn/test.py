import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import data
import model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='../data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=850,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=850,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=850,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.2,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--model_path', type=str,  default='EXP/model.pt',
                    help='path to load the pretrained model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=1,
                    help='number of experts')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
args = parser.parse_args()

def logging(s, print_=True, log_=True):
    print(s)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)


corpus = data.Corpus(args.data)
test_batch_size = 1
test_data = batchify(corpus.test, test_batch_size, args)


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        print(i, data_source.size(0)-1)
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

# Load the best saved model.
model = torch.load(args.model_path)

total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))
parallel_model = model.cuda()

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)

