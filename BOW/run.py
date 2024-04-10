# Author: Linus Lind
# Date: 21 March 2024 
# LICENSE: GNU GPLv3
###############################################################################
from time import perf_counter
start = perf_counter()
from os import path
from typing import NoReturn
import pandas as pd
import numpy as np

import preprocessing as pr
import model
import metrics

def main(traindata: pd.DataFrame,
         testdata: pd.DataFrame, *,
         gridsearch = False,
         recalculate_preprocessing = False) -> NoReturn:
    scaling = pr.minmaxscale

    # create copy to work with
    dataset_p = traindata.copy()
    testdata_p = testdata.copy()

    # preprocessing takes a long time, set recalculate_preprocessing = False if 
    # no changes to preprocessing have been made
    if recalculate_preprocessing:
        # apply preprocessing to text
        dataset_p['text'] = pr.process(traindata['text'])
        testdata_p['text'] = pr.process(testdata['text'])
        dataset_p.to_pickle(path.join(filepath, 'traindata_preprocessed.pkl'))
        testdata_p.to_pickle(path.join(filepath, 'testdata_preprocessed.pkl'))
    else:
        try:
            # load preprocessed data
            dataset_p = pd.read_pickle(path.join(filepath, 
                                                 'traindata_preprocessed.pkl'))
            testdata_p = pd.read_pickle(path.join(filepath, 
                                                  'testdata_preprocessed.pkl'))
        except FileNotFoundError:
            print(f'Preprocessed files traindata_preprocessed.csv',
                   'or testdata_preprocessed.csv not found')
            print(f'recalculating preprocessing')
            # apply preprocessing to text
            dataset_p['text'] = pr.process(traindata['text'])
            testdata_p['text'] = pr.process(testdata['text'])
            dataset_p.to_pickle(path.join(filepath, 
                                          'traindata_preprocessed.pkl'))
            testdata_p.to_pickle(path.join(filepath, 
                                           'testdata_preprocessed.pkl'))
    
    # separate train data to toxic and not toxic data
    toxic = dataset_p[dataset_p['label'].values == 1]
    not_toxic = dataset_p[dataset_p['label'].values == 0]
    # generate dictionaries and calculate probabilities
    toxic_prob = model.naive_probability(model.get_dict_counts(toxic))
    toxic_prob['vocab'] = toxic_prob.index.values
    toxic_prob.index.name = 'term'

    not_toxic_prob = model.naive_probability(model.get_dict_counts(not_toxic))
    not_toxic_prob['vocab'] = not_toxic_prob.index.values
    not_toxic_prob.index.name = 'term'

    # apply scaling to probabilities
    toxic_prob['prob'] = scaling(toxic_prob['prob'].values)
    not_toxic_prob['prob'] = scaling(not_toxic_prob['prob'].values)

    # save to csv
    not_toxic_prob\
        .sort_values(by='prob', ascending = False)\
        .to_csv(path.join(filepath, 'not_toxic_prob.csv'))
    toxic_prob\
        .sort_values(by='prob', ascending = False)\
        .to_csv(path.join(filepath, 'toxic_prob.csv'))
    
    thresholds = np.linspace(-0.1,0.1,10001)
    best_threshold = (0, -0.03984)
    f1 = 0
    best_preds = None
    scores = []

    results, tox, not_tox = model.compare(toxic_prob, 
                                          not_toxic_prob,
                                          testdata_p, 
                                          threshold=best_threshold[1])
    F_score_beta = 1

    real = testdata_p['label'].values
    # gridsearch = find optimal threshold value
    if gridsearch:
        for threshold in thresholds:
            preds = tox > (not_tox + threshold)
            acc = metrics.accuracy(preds, real)
            prec = metrics.precision(preds, real)
            rec = metrics.recall(preds, real)
            F = metrics.fx_score(preds, real, F_score_beta)
            if F > best_threshold[0]:
                best_threshold = (F, threshold)
                scores = [acc, prec, rec, F]
                f1 = metrics.f1_score(preds, real)
                best_preds = preds
    else:
        best_preds = results['preds'].values
        acc = metrics.accuracy(best_preds, real)
        prec = metrics.precision(best_preds, real)
        rec = metrics.recall(best_preds, real)
        F = metrics.fx_score(best_preds, real, F_score_beta)
        f1 = metrics.f1_score(best_preds, real)
        scores = [acc, prec, rec, F]

    print(f'Accuracy : {scores[0]}')
    print(f'Precision: {scores[1]}')
    print(f'Recall: {scores[2]}')
    print(f'F Score: {scores[3]}')
    print(f'F1 Score: {f1}')
    print(f'best threshold: {best_threshold[1]}')
    output = pd.DataFrame()
    output['preds_w_best_thres'] = best_preds
    output['toxic_p'] = tox
    output['not_toxic_p'] = not_tox
    output.to_csv(path.join(filepath, 'results.csv'))

if __name__ == "__main__":
    filepath = path.relpath('data')
    dev = pd.read_csv(path.join(filepath, "dev_2024.csv"), \
                      quoting=3, index_col = 'id')
    test = pd.read_csv(path.join(filepath, "test_2024.csv"), \
                       quoting=3, index_col = 'id')
    train = pd.read_csv(path.join(filepath, "train_2024.csv"), \
                        quoting=3, index_col = 'id')
    main(train, train, gridsearch=False, recalculate_preprocessing=True)
end = perf_counter()
runtime = end - start
print(f'Runtime of script: {runtime}')