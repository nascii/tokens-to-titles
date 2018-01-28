from time import time
import torch
import numpy as np
import pandas as pd
from langdetect import detect
# import matplotlib.pyplot as plt
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

np.random.seed(42)

nltk.download('punkt')
nltk.download('stopwords')
stemmer = LancasterStemmer()
en_stopwords = stopwords.words('english')
en_stopwords = [t for w in en_stopwords for t in nltk.word_tokenize(w)]
en_stopwords = [stemmer.stem(w) for w in en_stopwords]
en_stopwords = set(en_stopwords)

MAX_LENGTH = 20
SOS_token = 0
EOS_token = 1
use_cuda = torch.cuda.is_available()


# get_eng_film_titles('data/title.basics.tsv', 'data/title.ratings.tsv', 'data/en_titles.csv', 100000)

def get_eng_film_titles(titles_path, ratings_path, save_path, min_num_votes=100):
    titles = pd.read_table(titles_path, usecols=['tconst', 'primaryTitle'])
    ratings = pd.read_table(ratings_path)

    # Let's convert films ids into strings
    titles['id'] = titles.tconst.apply(lambda s: int(s[2:]))
    ratings['id'] = ratings.tconst.apply(lambda s: int(s[2:]))
    # print(titles[titles['id'].duplicated()].shape) # Checking that there are no duplicates
    # print(ratings[ratings['id'].duplicated()].shape) # Checking that there are no duplicates

    titles.set_index('id', inplace=True)
    ratings.set_index('id', inplace=True)

    titles.drop(['tconst'], axis=1, inplace=True)
    ratings.drop(['tconst'], axis=1, inplace=True)

    rated_films_ids = set(ratings[ratings.numVotes > min_num_votes].index.values)
    # rated_titles = titles[titles.index.isin(rated_films_ids)]
    rated_titles = titles[titles.index.isin(rated_films_ids)].copy()

    i = 0
    def detect_title_lang(title):
        global i
        i += 1
        if i % 100 == 0: print('Iterations done: {}/{}'.format(i, rated_titles.shape[0]))
        try:
            return detect(title)
        except KeyboardInterrupt:
            raise
        except:
            return 'unknown'

    start = time()
    rated_titles['lang'] = rated_titles.primaryTitle.apply(detect_title_lang)
    print('Took {:.1f} seconds'.format(time() - start))

    en_titles = rated_titles[rated_titles.lang == 'en'].primaryTitle.values

    print('Saving...')
    pd.DataFrame(en_titles, columns=['title']).save(save_path, index=None)


def load_en_titles(en_titles_path):
    en_titles = pd.read_csv(en_titles_path)
    return en_titles.title.values


def titles_to_pairs(titles):
    tokens = [[stemmer.stem(t) for t in nltk.word_tokenize(s)] for s in titles]
    tokens = [[t for t in s if not t in en_stopwords] for s in tokens]
    #tokens = [[t.encode('ascii', 'ignore').decode('ascii') for t in s] for s in tokens]

    data = pd.DataFrame({'inputs': tokens, 'targets': titles})

    # Dropping examples without tokens
    data['num_tokens'] = data.inputs.apply(lambda x: len(x))
    data = data[data.num_tokens != 0]

    tokenize_sentence = lambda s: ' '.join(nltk.word_tokenize(s))

    inputs = [' '.join(s) for s in data.inputs.values.tolist()]
    targets = [tokenize_sentence(s) for s in data.targets.values.tolist()]
    targets = [s.lower() for s in targets]

    pairs = [(inputs[i], targets[i]) for i in range(len(inputs))]

    return pairs
