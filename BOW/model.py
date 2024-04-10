# Author: Linus Lind
# Date: 21 March 2024 
# LICENSE: GNU GPLv3
###############################################################################
# Implements probability based bag-of-word model calculations
import numpy as np
import pandas as pd

# Calculate dict counts of each word
def get_dict_counts(df: pd.DataFrame) -> pd.DataFrame:
    word_dict = dict()
    for row in df['text'].values:
        for word in row:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:
                word_dict.update({word: word_dict.get(word) + 1})
    words = pd.DataFrame.from_dict(word_dict, orient='index')
    words.columns = ['counts']
    return words.sort_index()


# Sum probabilities
def calc_sum_probs(row: np.ndarray, 
                   df: pd.DataFrame) -> np.ndarray:
    out = 0
    for word in row:
        if word in df['vocab']:
            out += df['prob'].loc[word]
    return out / max(len(row), 1)

# Calculate probability according to dictionary
def naive_probability(count_dict: pd.DataFrame):
    N = len(count_dict)
    out = count_dict
    probs = np.log2(out['counts'].values) - np.log2(N)
    out.columns = ['prob']
    out['prob'] = probs
    return out

# Perform probability comparison according to toxic and not toxic dictionaries 
def compare(df_toxic: pd.DataFrame,
            df_not_toxic: pd.DataFrame,
            df_test: pd.DataFrame,
            *, threshold: float = 0.0) -> pd.DataFrame:
    out = pd.DataFrame()
    text = df_test['text']
    out['prob_tox'] = text.apply(lambda x: 
                                 calc_sum_probs(x, df_toxic))
    out['prob_not_tox'] = text.apply(lambda x: 
                                     calc_sum_probs(x, df_not_toxic))

    preds = out['prob_tox'].values > out['prob_not_tox'].values + threshold
    
    out['preds'] = np.where(preds, 1, 0)
    return out, out['prob_tox'].values, out['prob_not_tox'].values