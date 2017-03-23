#!/usr/bin/python3

"""

K-fold validation of Hidden markov model
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import sys
import POSTagger as pt
import numpy as np
import random

from sklearn import metrics
from sklearn.model_selection import KFold
from status import printProgressBar

def main(filename):
    kf = KFold(n_splits=5)
    f = pt.POSFile(filename)
    Lines = np.array(f.Lines)

    # shuffle the data.
    random.shuffle(Lines)

    for train_indices, test_indices in kf.split(Lines):
        train, test = Lines[train_indices], Lines[test_indices]
        test_len = len(test)
        print("len of train = ", len(train))
        print("len of test = ", test_len)
        h = pt.HMMTagger()
        h.Train(train)

        predictedTags = []
        expectedTags = []

        index = 0
        for line in test:
            sentence = line.Sentence
            words = sentence.split()
            if not words:
                continue
            predictedTags = h.Decode(sentence)

            assert len(predictedTags) == len(words),   \
                    "total tag sequence and len of words in sentence should be equal"

            expectedTags = [wt.tag for wt in line if not wt.IsFirstWord() and not wt.IsLastWord()]
            index += 1
            printProgressBar(index, test_len, prefix="Progress:", suffix="completed", length=50)

        print("Accuracy =", metrics.accuracy_score(expectedTags, predictedTags))
        print("===================")


if __name__ == "__main__":
    main(sys.argv[1])