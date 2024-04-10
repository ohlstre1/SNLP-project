# Author: Linus Lind
# Date: 21 March 2024 
# LICENSE: GNU GPLv3
###############################################################################
from os import path
import numpy as np
import pandas as pd
from nltk.data import find
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    find('corpora/stopwords')
except LookupError:
    download('stopwords')

try:
    find('corpora/wordnet.zip')
except LookupError:
    download('wordnet')

def remove_stopwords(row: list[str],
                     stopwords: set|list) -> list[str]:
    # Remove stopwords from the list of tokens
    filtered_tokens = [token for token in row \
                        if token not in stopwords and token.isalnum()]
    # Remove empty tokens
    out = [token for token in filtered_tokens if len(token) != 0]
    return out

# Apply lower casing, tokenization, stopword removal, lemmatization, store
# results as numpy array
def process(data: pd.Series) -> pd.Series:
    stemmer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    out = data.apply(str.lower)\
              .apply(word_tokenize)\
              .apply(lambda x: 
                     remove_stopwords(x, stop_words))\
              .apply(lambda x: [stemmer.lemmatize(token) for token in x])\
              .apply(np.array)
    
    return out


# Min-Max scaling for range [any, any] -> [0, 1]
def minmaxscale(arr: np.ndarray) -> np.ndarray:
    minval = np.min(arr)
    maxval = np.max(arr)
    diff = maxval - minval
    return (arr - minval) / diff