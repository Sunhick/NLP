#!/usr/bin/python3

"""
N-gram language model. 

Build a unigram and bigram model with and without smoothing(Add k smoothing).
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"

import re
import os
import sys
import math
from copy import deepcopy
from collections import defaultdict

class IncorrectFile(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class Ngram(object):
    # Makers the begining / end of sentence.
    SENTENCE_BEGIN = "<s>"
    SENTENCE_END = "</s>"

    # Constant for add-k smoothing
    K = .0001
    
    def __init__(self, trainFile):
        # When you access a key in a defaultdict, if it is not there, 
        # it will be created automatically. Since we have int as the 
        # default factory function, it creates the key and gives 
        # the default value 0.
        self.unigramsCounter = defaultdict(int)
        self.bigramsCounter = defaultdict(int)
        self.__constructModel(trainFile)

    def __constructModel(self, trainFile):
        # create the Ngram model. Count the occurances of 
        # unigram and bigram words. The tokens are not separated
        # on sentence boundary (as per write up).
        with open(trainFile, 'r') as file:
            for sentence in file:
                # make it case insensitive by converting to lower case.
                # strip the new lines either at begin or at the end of sentence.
                # extract words by splitting them on space delimiter.
                tokens = sentence.lower().strip().split()
                # create unigrams and bigrams
                unitokens = self.GetNgrams(deepcopy(tokens), 1)
                bitokens = self.GetNgrams(deepcopy(tokens), 2)
                # update the counters
                self.UpdateCounter(unitokens, self.unigramsCounter)
                self.UpdateCounter(bitokens, self.bigramsCounter)

    def GetNgrams(self, tokens, ngram):
        # begining of the sentence
        tokens.insert(0, self.SENTENCE_BEGIN)
        # end of sentence token
        tokens.append(self.SENTENCE_END)
        return zip(*[tokens[i:] for i in range(ngram)])

    def GetUnigramCounts(self, word):
        return self.unigramsCounter[word] if word in self.unigramsCounter else 0

    def GetBigramCounts(self, word):
        return self.bigramsCounter[word] if word in self.bigramsCounter else 0

    # Or GetUnigramAndBigramCounts()
    def GetUnigramProbabilities(self, word):
        # eliminate the sentence markers(-2)
        # L = len(unigramsCounter)-2
        L = sum(self.unigramsCounter.values()) - \
                    self.unigramsCounter[(self.SENTENCE_BEGIN,)] - self.unigramsCounter[(self.SENTENCE_END,)]
        prob = float(self.GetUnigramCounts(word)) / L
        return prob

    def GetBigramProbabilities(self, bigram):
        prob = 0.0
        first,_ = bigram
        prob = float(self.GetBigramCounts(bigram))/self.GetUnigramCounts((first,))
        return prob

    def GetSmoothedProbabilities(self, bigram, K):
        V = len(self.unigramsCounter)
        first, _ = bigram
        prob = float(self.GetBigramCounts(bigram)+K)/(self.GetUnigramCounts((first,))+V)
        return prob

    def GetUnigramSentenceProbability(self, sentence):
        prob = 0.0
        for word in self.GetNgrams(sentence.split(), 1):
            # Ignore sentence begin/end markers
            if word[0] == "<s>" or word[0] == "</s>":
                continue
            prob += math.log10(self.GetUnigramProbabilities(word))

        return prob

    def GetBigramSentenceProbability(self, sentence):
        prob = 0.0
        for bigram in self.GetNgrams(sentence.split(), 2):
            try:
                prob += math.log10(self.GetBigramProbabilities(bigram))
            except ValueError:
                # If p=0, log10(p) will throw exception. Log10(0) is undefined
                return 0
        return prob

    def GetSmoothedSentenceProbability(self, sentence, K):
        prob = 0.0
        for bigram in self.GetNgrams(sentence.split(), 2):
            # Probability of unseen word is not zero, but negligible
            prob += math.log10(self.GetSmoothedProbabilities(bigram, K))

        return prob

    def UpdateCounter(self, tokens, counter):
        for token in tokens:
                # increment the counter for this word
                counter[token] += 1
        return counter

def main(cmdline):
    trainingFile = cmdline[0]
    testFile = cmdline[1]

    model = Ngram(trainingFile)

    # Constant for add-k smoothing
    K = .0001

    # Test the Ngram model on test file and calculate the 
    # sentence prediction probability
    with open(testFile, 'r') as testFile:
        for line in testFile:
            sentence = line.lower().strip()
            uniProb = model.GetUnigramSentenceProbability(sentence)
            biProb = model.GetBigramSentenceProbability(sentence)
            biSmoothProb = model.GetSmoothedSentenceProbability(sentence, K)

            print("S =", line.strip())
            print("Unigrams: logprob(S) =", "{:.4f}".format(uniProb))
            print("Bigrams: logprob(S) =", "undefined" if biProb==0 else "{:.4f}".format(biProb))
            print("Smoothed Bigrams: logprob(S) =", "{:.4f}".format(biSmoothProb))
            print()

if __name__ == '__main__':
    raise IncorrectFile("Don't use this file! use the latest file ngram.py")
    # main(sys.argv[1:])