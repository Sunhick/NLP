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

import os
import POSTagger as pt

def main():
    f = pt.POSFile("berp-POS-train.txt")

    for i in range(1, 30):
        train, test = f.RandomSplit(80)

        formatter = lambda word, tag: "{0}\t{1}\n".format(word, tag)
        endOfSentence = ".\t.{newline}{newline}".format(newline=os.linesep)
        testEndOfSentence = ".{newline}{newline}".format(newline=os.linesep)

        with open("data/train_{0}.txt".format(i), "w") as file:
            for line in train:
                sentence = line.Sentence
                words = sentence.split()
                if not words:
                    continue

                for wt in line:
                    if not wt.IsFirstWord() and not wt.IsLastWord():
                        w, t = wt
                        file.write(formatter(w, t))
                file.write(endOfSentence)

        noTagformatter = lambda word: "{0}\n".format(word)
        with open("data/test_{0}.txt".format(i), "w") as file,     \
             open("data/berp-key_{0}.txt".format(i), "w") as goldFile:
            for line in test:
                sentence = line.Sentence
                words = sentence.split()
                if not words:
                    continue

                for wt in line:
                    if not wt.IsFirstWord() and not wt.IsLastWord():
                        w, t = wt
                        file.write(noTagformatter(w))
                file.write(testEndOfSentence)

                for wt in line:
                    if not wt.IsFirstWord() and not wt.IsLastWord():
                        w, t = wt
                        goldFile.write(formatter(w, t))
                goldFile.write(endOfSentence)


if __name__ == "__main__":
    main()