
# Hate Speech Detection for Tweets with Surface-level Features

In this project, we detect whether tweets contain hate speech (offensive words, implications, slang, etc.). The problem can prove challenging due to several reasons such as:

  - Tweets contain many out-of-dictionary words (e.g. gr8), slang abbreviations (e.g. smh) and hashtags (e.g. #CorruptPolitician).
  - Tweets contain emojis, which strongly relate to the sentiment. 
  - Tweets are unstructured. They can contain one or more sentences.

We are required to develop a system that uses surface-level features (i.e. not a neural net) to outperform a given baseline classifier.

In our approach, we combine several features used in the literature to form the feature vector and use SVM as a classifier. 

Our approach reaches accuracy as high as 72%, blind on a given testset.



## File Description

The project contains the following files:

   1. Code:
       - `main.py`: main code. Runs feature extraction and train the model on trainset.
       - `feature_extraction.py`: extracts features from tweets. Called in `main.py`.
       - `download_data.py`: Downloads the necessary data files from our git repo.
       - `emoji_split.py`: a small script that splits multiple emojis in the same tweet. We used this before feeding the data to TwitterNLP tool.
2. Data
    - `nrc_unigrams`: unigram sentiment scores from NRC dataset.
    - `nrc_bigrams`: bigram sentiment scores from NRC dataset.
    - `sentiment140_unigrams`: unigram sentiment scores from Sentiment140 dataset.
    - `sentiment140_bigrams`: bigram sentiment scores from Sentiment140 dataset.
    - `word_clusters`: mapping from words to 1000 clusters provided by CMU TweetNLP tool.
    - `train_final_out`: trainset tokenized by the CMU TweetNLP tool.
    - `train_final_out_pos`: trainset pos-tagged by the CMU TweetNLP tool.
    - `dev_final_out`: devset tokenized by the CMU TweetNLP tool.
    - `dev_final_out_pos`: devset pos-tagged by the CMU TweetNLP tool.
    - `test_final_out`: testset tokenized by the CMU TweetNLP tool.
    - `test_final_out_pos`: testset pos-tagged by the CMU TweetNLP tool.
3. Output
    - `predictions.test`: predictions of the test set evaluated on the improved model.

## How To Run

1. Download the necessary data for feature extraction
    `python download_data.py`

2. Run the code to train on train, and give train, dev accuracy and test predictions as test_predictions.tsv
    `python main.py`

### Libraries Required:

- Python 3
- Scikit Learn
- Pandas
- Numpy
    


