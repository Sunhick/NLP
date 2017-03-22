#!/usr/bin/python3

"""

Gather statistics about POS tagged file using Hidden markov model.

Statistics:
  * Top N miscategorized tags
  * Show confusion matrix for all tags
"""

__author__ = "Sunil"
__copyright__ = "Copyright (c) 2017 Sunil"
__license__ = "MIT License"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"

import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

from POSTagger import POSFile

from sklearn import metrics

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    """confusion matrix

    confusion matrix usage to evaluate the quality of the output of a classifier on the image data set. 
    The diagonal elements represent the number of points for which the predicted label is equal to the 
    true label, while off-diagonal elements are those that are mislabeled by the classifier. 
    The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
    """
    np.set_printoptions(precision=2)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plotcnf(expectedTags, predictedTags):
  cnf_matrix = metrics.confusion_matrix(expectedTags, predictedTags)
  class_names = list(set(expectedTags))

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, title="Confusion matrix")

def main(args):
  goldFilename = args[1]
  outFilename = args[2]

  gdata = POSFile(goldFilename)
  odata = POSFile(outFilename)

  assert len(gdata.Lines) == len(odata.Lines), "gold file and output file have different line count!"

  expectedTags = [wt.tag for e in gdata.Lines for wt in e.words           \
                    if not wt.IsFirstWord()]

  predictedTags = [wt.tag for e in odata.Lines for wt in e.words           \
                    if not wt.IsFirstWord()]

  # plotcnf(expectedTags, predictedTags)

  print("Accuracy =", metrics.accuracy_score(expectedTags, predictedTags))
  print("Precision =", metrics.precision_score(expectedTags, predictedTags, average='weighted'))
  # recall throws warning, because FP = 0 and we end up 0/0
  # print("Recall =", metrics.recall_score(expectedTags, predictedTags, average='weighted'))

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Incorrect number of arguments!")
    print("usage: python3 <berp-key.txt> <berp-out.txt>")
    sys.exit(0)
  main(sys.argv)