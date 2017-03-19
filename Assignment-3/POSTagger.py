#!/usr/bin/python3

"""
POS tagging using HMM (Viterbi algorithm)
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"

import os
import sys
import math

# not required for now. as we are not splitting contractions ourself.
# but requried when testing with sentences.
# from nltk.tokenize import *

from collections import defaultdict

tagTransitions = defaultdict(lambda: defaultdict(float))
likelihood = defaultdict(lambda: defaultdict(float))

class WordTag(object):
    """
    Represents the word tag pair.
    """
    word = tag = None

    def __init__(self, word, tag):
        self.word = word
        self.tag = tag

class Line(object):
    """
    Represents the sentence as collection of words and thier 
    corresponding tag sequences.
    """
    words = []

    def __init__(self):
        self.words = []

    def AddWordTag(self, word, tag):
        wordTag = WordTag(word, tag)
        self.words.append(wordTag)

class File(object):
    """
    Abstraction over words-tags, lines as a file
    """
    lines = []

    def __init__(self, filename):
        self.__read(filename)

    def __read(self, filename):
        sentence = Line()
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip():
                    # end of sentence
                    self.lines.append(sentence)
                    sentence = Line()
                    continue

                word, tag = line.split()
                # print("word=", word, "tag=", tag)
                sentence.AddWordTag(word, tag)

    @property
    def Lines(self):
        return self.lines

    def Split(self, train_percent):
        size = len(self.lines)
        train_len = int(size*train_percent/100)
        test_len = size - train_len
        return self.lines[:train_len], self.lines[-test_len:]

    def RandomSplit(self, train_percent):
        lines = deepcopy(self.lines)
        random.shuffle(lines)
        size = len(lines)
        train_len = int(size*train_percent/100)
        test_len = size - train_len
        return lines[:train_len], lines[-test_len:]

def train(filename):
    lines = []
    lc = 0
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                # end of sentence
                lc += 1
                continue
            # print(line.split())
            word, tag = line.split()
            
            if word == "." and tag == ".":
                continue
            likelihood[tag][word] += 1
    print(lc)

def main(args):
    filename = args[0]
    train(filename)

if __name__ == "__main__":
    main(sys.argv[1:])