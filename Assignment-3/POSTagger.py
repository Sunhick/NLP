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

# to create bigram partial class from ngram
# from functools import partial
# Bigrams = partial(Ngrams, n = 2)

from collections import defaultdict
from collections import namedtuple

kSENTENCE_BEGIN = "<s>"
kSENTENCE_END = "</s>"

class WordTag(namedtuple('WordTag', ['word', 'tag'], verbose=False)):
    """
    Represents the word tag pair. Inherits from namedtuple so that i can 
    unpack the class in word, tag pair easily in the iterations.
    """
    word = tag = None

    def __init__(self, word, tag):
        # ignore the case of word
        self.word = word.lower()
        self.tag = tag

    def IsLastWord(self):
        return self.word == kSENTENCE_END

    def IsFirstWord(self):
        return self.word == kSENTENCE_BEGIN

class Line(object):
    """
    Represents the sentence as collection of words and thier 
    corresponding tag sequences.
    """
    words = []
    __index = 0

    def __init__(self):
        self.words = []

    def AddWordTag(self, word, tag):
        wordTag = WordTag(word, tag)
        self.words.append(wordTag)

    def __iter__(self):
        # called once before iteration. reset index pointer
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.words):
            raise StopIteration
        else:
            self.__index += 1
            return self.words[self.__index-1]

class File(object):
    """
    Abstraction over words-tags, lines as a file
    """
    lines = []

    def __init__(self, filename):
        self.__read(filename)

    def __read(self, filename):
        sentence = Line()
        sentence.AddWordTag(kSENTENCE_BEGIN, kSENTENCE_BEGIN)
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip():
                    # end of sentence
                    self.lines.append(sentence)

                    # create a new line holder
                    sentence = Line()
                    # add the word begin marker
                    sentence.AddWordTag(kSENTENCE_BEGIN, kSENTENCE_BEGIN)
                    continue

                word, tag = line.split()

                # Marks the last word in the sentence
                if word == "." and tag == ".":
                    # add the word end marker
                    sentence.AddWordTag(kSENTENCE_END, kSENTENCE_END)
                else:
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

class Ngrams(object):
    """
    Generates the ngrams from the list of words
    """
    Ngrams = []
    __index = 0

    def __init__(self, words, n):
        self.Ngrams = list(zip(*[words[i:] for i in range(n)]))

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self):
        if self.__index >= len(self.Ngrams):
            raise StopIteration
        else:
            self.__index += 1
            return self.Ngrams[self.__index-1]

class Bigrams(Ngrams):
    def __init__(self, words):
        super(Bigrams, self).__init__(words, n = 2)

class Viterbi(object):
    def __call__(self, tagger, sentence):
        tagSequence = []
        # implement viterbi algorithm here

        # follow the back pointers to get a tag sequence.
        return tagSequence

class HMMTagger(object):
    """
    POS tagger using HMM. Each word may have more tags assosicated with it.
    """
    tagTransitions = likelihood = None
    V = 0       # vocabulary size. Total count of tags 
    k = 0.0001  # maybe i have to fine tune this to get better accuracy.
    __decoder = None

    def __init__(self, decoder = Viterbi()):
        self.tagTransitions = defaultdict(lambda: defaultdict(float))
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.__decoder = decoder

    def Train(self, trainData):
        for line in trainData:
            # update the likelihood probabilities
            for word, tag in line:
                # unpack word and tag. I can do this becusae of namedtuple
                self.likelihood[tag][word] += 1

            words = line.words
            # update the tag transition probabilties
            for first, second in Bigrams(words).Ngrams:
                _, fromTag = first
                _, toTag = second
                self.tagTransitions[fromTag][toTag] += 1

        # Normalize probablities
        self.__normalize()

    def __normalize(self):
        # -2 because of start and end sentence markers
        # -1 because of defaultdict's default value when called with no parameters
        self.V = len(set(self.tagTransitions)) - 2 - 1

        # If i normalize the tag transition table, 
        # I can directly use it and no need for 
        # below two methods.
        pass

    def GetTagTransitionProbability(self, fromTag, toTag):
        """
                      C(X, Y) + k
        P(Y | X) =   ______________
                        C(X) + Vk

        Use add-k with k = 0.0001 ? (assignment-2 value)
        """
        prob = 0.0
        cxy = self.tagTransitions[fromTag][toTag]
        cx = sum(self.tagTransitions[fromTag].values)
        prob = (cxy + self.k) / (cx + (self.k * self.V))
        return prob

    def GetLikelihoodProbability(self, tag, word):
        """
                            C(tag, word) + k
        P(word | tag) =    ___________________
                              C(tag) + Vk

        Use add-k with k = 0.0001 ? (assignment-2 value)
        """
        prob = 0.0
        ctagword = self.likelihood[tag][word]
        ctag = sum(self.likelihood[tag].values())
        prob = (ctagword + self.k) / (ctag + (self.k * self.V))


    def Decode(self, sentence):
        return self.__decoder(self, sentence)

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
    # train(filename)
    data = File(filename)

    # split data as 80:20 for train and test 
    train, test = data.Split(80)
    tagger = HMMTagger(Viterbi())
    tagger.Train(train)

if __name__ == "__main__":
    main(sys.argv[1:])