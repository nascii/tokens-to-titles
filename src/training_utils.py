import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.data_utils import titles_to_pairs


teacher_forcing_ratio = 0.5
MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()



def train(input_variable, target_variable, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input


            loss += criterion(decoder_output, target_variable[di])

            if ni == EOS_token:
                break

    loss.backward()
    optimizer.step()

    return loss.data[0] / target_length


def word_to_index(lang, word, unknown_noise_prob=0.1):
    if word in lang.word2index and random.random() > unknown_noise_prob:
        return lang.word2index[word]
    else:
        return lang.word2index['UNKNOWN']


def tokenize_sentence(lang, sentence):
    pair = titles_to_pairs([sentence])[0]
    return pair[0]


def sentence_to_index(lang, sentence):
    return [word_to_index(lang, word) for word in sentence.split(' ')]


def sentence_to_var(lang, sentence):
    indexes = sentence_to_index(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))

    return result.cuda() if use_cuda else result


def pair_to_var(pair, source_lang, target_lang):
    input_variable = sentence_to_var(source_lang, pair[0])
    target_variable = sentence_to_var(target_lang, pair[1])

    return (input_variable, target_variable)


def evaluate(encoder, decoder, source_lang, target_lang, sentence, max_length=MAX_LENGTH):
    input_variable = sentence_to_var(source_lang, tokenize_sentence(source_lang, sentence))
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(target_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words[:-1]
