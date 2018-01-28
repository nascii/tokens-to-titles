import sys
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from src.lang import Lang
from src.training_utils import train, evaluate, pair_to_var
from src.data_utils import load_en_titles, titles_to_pairs
from src.models import Encoder, Decoder


hidden_size = 128


def generate_langs(inputs, targets):
    source_lang = Lang('source')
    target_lang = Lang('target')

    for s in inputs: source_lang.add_sentence(s)
    for s in targets: target_lang.add_sentence(s)

    return source_lang, target_lang


def load_dataset():
    titles = load_en_titles('data/en_titles.csv')
    pairs = titles_to_pairs(titles)

    inputs = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]

    return inputs, targets, pairs


def load_inference():
    inputs, targets, pairs = load_dataset()
    source_lang, target_lang = generate_langs(inputs, targets)

    encoder = Encoder(hidden_size, source_lang.n_words)
    decoder = Decoder(hidden_size, target_lang.n_words)

    encoder.load_state_dict(torch.load('models/encoder.state_dict.pth'))
    decoder.load_state_dict(torch.load('models/decoder.state_dict.pth'))

    def run_model(sentence):
        sentence = sentence.strip()
        output = evaluate(encoder, decoder, source_lang, target_lang, sentence)
        output = ' '.join(output)
        output = output[0].upper() + output[1:]

        return output

    return run_model


def run_training(num_epochs=5):
    learning_rate = 1e-3

    inputs, targets, pairs = load_dataset()
    source_lang, target_lang = generate_langs(inputs, targets)

    encoder = Encoder(hidden_size, source_lang.n_words)
    decoder = Decoder(hidden_size, target_lang.n_words)

    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ], lr=learning_rate)

    criterion = nn.NLLLoss()
    losses = []

    print('Running training for {} epochs'.format(num_epochs))

    for epoch in range(num_epochs):
        loss = 0

        shuffled_pairs = random.sample(pairs, len(pairs))

        for _, pair in tqdm(enumerate(pairs), total=len(pairs)):
            input_var, target_var = pair_to_var(pair, source_lang, target_lang)
            loss += train(input_var, target_var, encoder, decoder, optimizer, criterion)

        losses.append(loss)
        print('Epoch #{}. Loss: {:.2f}'.format(epoch+1, loss))
        
        print('Saving models...')
        torch.save(encoder.state_dict(), 'models/encoder.state_dict.pth')
        torch.save(decoder.state_dict(), 'models/decoder.state_dict.pth')
        print('Models saved')

    print('Losses:', losses)



def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print('Traininig')
        run_training()
    else:
        print('Other options are not supported')


if __name__ == '__main__':
    main()
