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

# When you access a key in a defaultdict, if it is not there, 
# it will be created automatically. Since we have int as the 
# default factory function, it creates the key and gives 
# the default value 0.
unigramsCounter = defaultdict(int)
bigramsCounter = defaultdict(int)

# Makers the begining / end of sentence.
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

# Constant for add-k smoothing
K = .0001

def GetNgrams(tokens, ngram):
    # begining of the sentence
    tokens.insert(0, SENTENCE_BEGIN)
    # end of sentence token
    tokens.append(SENTENCE_END)
    return zip(*[tokens[i:] for i in range(ngram)])
    

def GetUnigramCounts(word):
    return unigramsCounter[word] if word in unigramsCounter else 0

def GetBigramCounts(word):
    return bigramsCounter[word] if word in bigramsCounter else 0

# Or GetUnigramAndBigramCounts()
def GetUnigramProbabilities(word):
    prob = float(GetUnigramCounts(word)) / len(unigramsCounter)
    return prob

def GetBigramProbabilities(bigram):
    prob = 0.0
    first,_ = bigram
    prob = float(GetBigramCounts(bigram))/GetUnigramCounts((first,))
    return prob

def GetSmoothedProbabilities(word):
    V = len(unigramsCounter)
    prob = float(GetBigramCounts(word)+K)/(GetUnigramCounts(word)+V)
    return prob

# Or GetBigramAndSmoothedProb()
def GetUnigramSentenceProbability(sentence):
    prob = 0.0
    for word in GetNgrams(sentence.split(), 1):
        # Ignore sentence begin/end markers
        if word[0] == "<s>" or word[0] == "</s>":
            continue
        prob += math.log10(GetUnigramProbabilities(word))

    return prob

def GetBigramSentenceProbability(sentence):
    prob = 0.0
    for bigram in GetNgrams(sentence.split(), 2):
        p = GetBigramProbabilities(bigram)
        if p == 0:
            # if probability of one of the word is zero, 
            # then the probability of entire sentence is zero.
            # Probability of unseen word is zero
            return 0
        prob += math.log10(p)
    return prob

def GetSmoothedSentenceProbability(sentence):
    prob = 0.0
    for word in GetNgrams(sentence.split(), 2):
        # Probability of unseen word is not zero, but negligible
        prob += math.log10(GetSmoothedProbabilities(word))

    return prob

def UpdateCounter(tokens, counter):
    for token in tokens:
            # increment the counter for this word
            counter[token] += 1
    return counter

def main(cmdline):
    # use the global variables 
    global unigramsCounter
    global bigramsCounter
    trainingFile = cmdline[0]
    testFile = cmdline[1]

    # create the Ngram model. Count the occurances of 
    # unigram and bigram words. The tokens are not separated
    # on sentence boundary (as per write up).
    with open(trainingFile, 'r') as trainFile:
        for sentence in trainFile:
            tokens = sentence.lower().strip().split()
            #if lastword:
            #    tokens.insert(0, lastword)
            unitokens = GetNgrams(deepcopy(tokens), 1)
            bitokens = GetNgrams(deepcopy(tokens), 2)
            #lastword = tokens[-1]
            unigramsCounter = UpdateCounter(unitokens, unigramsCounter)
            bigramsCounter = UpdateCounter(bitokens, bigramsCounter)

    # print(len(unigramsCounter))
    # print(len(bigramsCounter))

    # Now use the test file to calculate probabilities
    # What should i do now. read the test file and calculate 
    # unigram and bigram model probabilities for each sentence?

    # Test the Ngram model on test file and calculate the 
    # sentence prediction probability
    with open(testFile, 'r') as testFile:
        for sentence in testFile:
            sentence = sentence.lower().strip()
            uniProb = GetUnigramSentenceProbability(sentence)
            biProb = GetBigramSentenceProbability(sentence)
            biSmoothProb = GetSmoothedSentenceProbability(sentence)

            print("S = ", sentence)
            print("Unigrams: logprob(S) = ", uniProb)
            print("Bigrams: logprob(S) = ", biProb)
            print("Bigrams Smoothing: logprob(S) = ", biSmoothProb)
            print()

if __name__ == '__main__':
    main(sys.argv[1:])