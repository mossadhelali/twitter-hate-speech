import numpy as np
import pandas as pd
import csv
import re
from collections import OrderedDict
from operator import add


################################################
print('preparing feature extraction module ...')
# preparation steps. This code runs only once
# 1. prepare lexicon datasets
    # read the files
nrc_unigrams = pd.read_csv('nrc_unigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
nrc_bigrams = pd.read_csv('nrc_bigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
sentiment140_unigrams = pd.read_csv('sentiment140_unigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)
sentiment140_bigrams = pd.read_csv('sentiment140_bigrams.tsv', sep='\t', names=['word','score','npos', 'nneg'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)

    # drop extra columns and convert to dict, where key is uni/bigram and value is the score (whehter it is postive or negative). NOTE: takes some time (< 1 min)
nrc_unigrams = nrc_unigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')
nrc_bigrams = nrc_bigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')
sentiment140_unigrams = sentiment140_unigrams.drop(['npos','nneg'],axis=1)[~sentiment140_unigrams.duplicated(subset='word', keep='first')].set_index('word').to_dict(orient='index')
sentiment140_bigrams = sentiment140_bigrams.drop(['npos','nneg'],axis=1).set_index('word').to_dict(orient='index')

for k, v in nrc_unigrams.items():
    nrc_unigrams[k] = v['score']

for k, v in nrc_bigrams.items():
    nrc_bigrams[k] = v['score']

for k, v in sentiment140_unigrams.items():
    sentiment140_unigrams[k] = v['score']

for k, v in sentiment140_bigrams.items():
    sentiment140_bigrams[k] = v['score']

# read word clusters. convert into a dict where key is word and value is cluster. Also keep list of all clusters.
word_clusters = pd.read_csv('word_clusters.tsv', sep='\t', names=['cluster','word','n_occur'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['n_occur'],axis=1)
cluster_set = list(word_clusters['cluster'].unique())
word_clustres = word_clusters[~word_clusters.duplicated(subset='word', keep='first')].set_index('word')
word_clusters = word_clusters.to_dict(orient='index')

for k, v in word_clusters.items():
    word_clusters[k] = v['cluster']


# gather all character 3-grams from train and dev sets.
tweets_train = pd.read_csv('train_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)['tweet'].str.cat(sep=' ')
tweets_dev = pd.read_csv('dev_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False)['tweet'].str.cat(sep=' ')


three_grams = set()

for word in tweets_train.split():

    three_grams.update([word[i:i+3] for i in range(len(word)-3+1)])

for word in tweets_dev.split():

    three_grams.update([word[i:i+3] for i in range(len(word)-3+1)])


# prepare glove representations
glove200 = {}
with open('glove_200.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line_split = line.split()
        glove200[line_split[0]] = [float(i) for i in line_split[1:]]

###########################################################


# functions to extract each group of features
# each takes a list of words and returns a feature vector

def get_lexicon_features(sent):
    # lexicon features are for unigrams and bigrams from two different reference datasets: NRC and Sentiment140

    # 1. get the score for each uni/bigram in the tweet
    nrc_unigram_scores = []
    nrc_bigram_scores = []
    sentiment140_unigram_scores = []
    sentiment140_bigram_scores = []

    for word in sent:
        unigram = word.lower()
        if unigram in nrc_unigrams:
            nrc_unigram_scores.append(nrc_unigrams[unigram])
        if unigram in sentiment140_unigrams:
            sentiment140_unigram_scores.append(sentiment140_unigrams[unigram])

    for i in range(len(sent)-1):
        bigram = sent[i].lower() + ' ' + sent[i+1].lower()
        if bigram in nrc_bigrams:
            nrc_bigram_scores.append(nrc_bigrams[bigram])
        if bigram in sentiment140_bigrams:
            sentiment140_bigram_scores.append(sentiment140_bigrams[bigram])

    # 2. compute the feature vectors based on the scores.
    nrc_unigram_features = []
    nrc_bigram_features = []
    sentiment140_unigram_features = []
    sentiment140_bigram_features = []

        # a. num of scores > 0
    nrc_unigram_features.append(len([s for s in nrc_unigram_scores if s > 0]))
    nrc_bigram_features.append(len([s for s in nrc_bigram_scores if s > 0]))
    sentiment140_unigram_features.append(len([s for s in sentiment140_unigram_scores if s > 0]))
    sentiment140_bigram_features.append(len([s for s in sentiment140_bigram_scores if s > 0]))

        # b. num of scores < 0
    nrc_unigram_features.append(len([s for s in nrc_unigram_scores if s < 0]))
    nrc_bigram_features.append(len([s for s in nrc_bigram_scores if s < 0]))
    sentiment140_unigram_features.append(len([s for s in sentiment140_unigram_scores if s < 0]))
    sentiment140_bigram_features.append(len([s for s in sentiment140_bigram_scores if s < 0]))

        # c. sum of scores
    nrc_unigram_features.append(sum(nrc_unigram_scores))
    nrc_bigram_features.append(sum(nrc_bigram_scores))
    sentiment140_unigram_features.append(sum(sentiment140_unigram_scores))
    sentiment140_bigram_features.append(sum(sentiment140_bigram_scores))

        # d. sum of scores < 0
    nrc_unigram_features.append(sum([s for s in nrc_unigram_scores if s < 0]))
    nrc_bigram_features.append(sum([s for s in nrc_bigram_scores if s < 0]))
    sentiment140_unigram_features.append(sum([s for s in sentiment140_unigram_scores if s < 0]))
    sentiment140_bigram_features.append(sum([s for s in sentiment140_bigram_scores if s < 0]))

        # e. max score. if there are no scores at all, then 0. -10 is the lowest possible score.
    nrc_unigram_features.append(max([-10 * len(nrc_unigram_scores)] + nrc_unigram_scores))
    nrc_bigram_features.append(max([-10 * len(nrc_bigram_scores)] + nrc_bigram_scores))
    sentiment140_unigram_features.append(max([-10 * len(sentiment140_unigram_scores)] + sentiment140_unigram_scores))
    sentiment140_bigram_features.append(max([-10 * len(sentiment140_bigram_scores)] + sentiment140_bigram_scores))

        # f. min score. if there are no scores at all, then 0. -10 is the lowest possible score.
    nrc_unigram_features.append(min([10 * len(nrc_unigram_scores)] + nrc_unigram_scores))
    nrc_bigram_features.append(min([10 * len(nrc_bigram_scores)] + nrc_bigram_scores))
    sentiment140_unigram_features.append(min([10 * len(sentiment140_unigram_scores)] + sentiment140_unigram_scores))
    sentiment140_bigram_features.append(min([10 * len(sentiment140_bigram_scores)] + sentiment140_bigram_scores))


        # g. last score that is > 0. if no score > 0, then 0.
    nrc_unigram_features.append(([0]+[s for s in nrc_unigram_scores if s > 0])[-1])
    nrc_bigram_features.append(([0]+[s for s in nrc_bigram_scores if s > 0])[-1])
    sentiment140_unigram_features.append(([0]+[s for s in sentiment140_unigram_scores if s > 0])[-1])
    sentiment140_bigram_features.append(([0]+[s for s in sentiment140_bigram_scores if s > 0])[-1])

    # 3. return all features stacked
    return nrc_unigram_features + nrc_bigram_features + sentiment140_unigram_features + sentiment140_bigram_features


def get_punctuation_features(sent):
    # num of continguous sequences of ? or ! in the tweet
    num_contiguous_marks = 0
    for w in sent:
        finds = re.findall(r'((\?|\!){2,})', w)
        num_contiguous_marks += len(finds)

    # whether the last word contains ? or !
    last_contains_mark = 1 if '?' in sent[-1] or '!' in sent[-1] else 0

    return [num_contiguous_marks, last_contains_mark]

def get_all_caps_features(sent):
    # num of words with all caps
    all_caps_counter = 0
    for word in sent:
        if word != '@USER' and word == word.upper():
            all_caps_counter += 1

    return [all_caps_counter]

def get_elongated_words_features(sent):
    # num of words that are elongated. e.g. soooo
    elongated_words_counter = 0
    for word in sent:
        if re.search(r'(\w)\1\1', word):
            elongated_words_counter += 1

    return [elongated_words_counter]

def get_pos_features(pos_tags):
    # num of occurrence of each pos tag
    # set of possible tags. see: http://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf
    possible_tags = ['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P', '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G']
    tag_counts = []
    for tag in possible_tags:
        tag_counts.append(pos_tags.count(tag))
    return tag_counts


def get_word_cluster_features(sent):
    # occurence of each of the ~1000 word clusters provided by the CMU NLP tool
    cluster_presence = OrderedDict({cluster : 0 for cluster in cluster_set})   # ordered dict to preserve the order of the feature vector
    for word in sent:
        if word.lower() in word_clusters:
            cluster = word_clusters[word.lower()]
            cluster_presence[cluster] = 1

    return list(cluster_presence.values())


def get_character_three_gram_features(sent):
    # the presence or absence of each of the character 3-grams (~15.4K)
    n = 3
    three_gram_presence = OrderedDict({three_gram : 0 for three_gram in three_grams})   # ordered dict to preserve the order of the feature vector
    for word in sent:
        word_grams = [word[i:i+n] for i in range(len(word)-n+1)]
        for gram in word_grams:
            if gram in three_gram_presence:
                three_gram_presence[gram] = 1

    return list(three_gram_presence.values())


def get_glove200_features(sent):
    # average glove representation
    representation = [i*0 for i in range(200)]
    for word in sent:
        if word.lower() in glove200:
            representation = list(map(add, representation, glove200[word.lower()]))     # add two lists element-wise

    return representation #[i / len(sent) for i in representation]


def extract_all_features(tweet_row):
    sent = tweet_row['tweet'].split()
    pos_tags = tweet_row['pos']
    features = []
    # lexicon features
    features.extend(get_lexicon_features(sent))
    # word structure features
    features.extend(get_punctuation_features(sent))
    features.extend(get_all_caps_features(sent))
    features.extend(get_elongated_words_features(sent))
    # pos features
    features.extend(get_pos_features(pos_tags))
    # word cluster features
    features.extend(get_word_cluster_features(sent))
    # character n-gram features
    #features.extend(get_character_three_gram_features(sent))
    # glove features
    features.extend(get_glove200_features(sent))

    return features
