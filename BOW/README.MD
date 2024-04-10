# Author: Linus Lind
# Date: 21 March 2024 
# LICENSE: GNU GPLv3
# Toxicity detection with probability based Bag-Of-Words model
# metrics.py: methods accuracy, precision, recall, F score
# model.py: methods for creating bag-of-words dictionary with probabilities and comparing them to make predictions
# preprocessing.py: preprocessing methods. Use process() method to apply everything
# run.py: main file to run, use gridsearch=True to optimize threshold value
# For plug and play usage, have data set in data folder formatted as csv. sep = ',' with headers: id,text,label
# id = unique identifier
# text = text
# label = 0 or 1, (corresponding to not toxic or toxic)