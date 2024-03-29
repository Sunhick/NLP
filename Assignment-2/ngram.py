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
    # compute the total word frequency except sentence begin / end markers
    L = sum(unigramsCounter.values()) - unigramsCounter[(SENTENCE_BEGIN,)] - unigramsCounter[(SENTENCE_END,)]
    return float(GetUnigramCounts(word)) / L

def GetBigramProbabilities(bigram):
    first,_ = bigram
    return float(GetBigramCounts(bigram))/GetUnigramCounts((first,))

def GetSmoothedProbabilities(bigram):
    V = len(unigramsCounter)
    first, _ = bigram
    return float(GetBigramCounts(bigram)+K)/(GetUnigramCounts((first,))+(K*V))

# Or GetBigramAndSmoothedProb()
def GetUnigramSentenceProbability(sentence):
    prob = 0.0
    for word in GetNgrams(sentence.split(), 1):
        # Ignore sentence begin/end markers
        if word[0] == SENTENCE_BEGIN or word[0] == SENTENCE_END:
            continue
        try:
            prob += math.log10(GetUnigramProbabilities(word))
        except Exception:
            return 0

    return prob

def GetBigramSentenceProbability(sentence):
    prob = 0.0
    for bigram in GetNgrams(sentence.split(), 2):
        try:
            prob += math.log10(GetBigramProbabilities(bigram))
        except Exception:
            # If p=0, log10(p) will throw exception. Log10(0) is undefined
            return 0

    return prob

def GetSmoothedSentenceProbability(sentence):
    prob = 0.0
    for bigram in GetNgrams(sentence.split(), 2):
        try:
            # Probability of unseen word is not zero, but negligible
            prob += math.log10(GetSmoothedProbabilities(bigram))
        except Exception as e:
            return 0

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

    if len(cmdline) < 2:
        print("Missing training and test file.")
        print("$ python3 ngram.py trainingFile testFile")
        return

    trainingFile = cmdline[0]
    testFile = cmdline[1]

    # create the Ngram model. Count the occurances of 
    # unigram and bigram words. The tokens are not separated
    # on sentence boundary (as per write up).
    with open(trainingFile, 'r') as trainFile:
        for sentence in trainFile:
            # make it case insensitive by converting to lower case.
            # strip the new lines either at begin or at the end of sentence.
            # extract words by splitting them on space delimiter.
            tokens = sentence.lower().strip().split()
            
            # create unigrams and bigrams
            unitokens = GetNgrams(deepcopy(tokens), 1)
            bitokens = GetNgrams(deepcopy(tokens), 2)
            
            # update the counters
            UpdateCounter(unitokens, unigramsCounter)
            UpdateCounter(bitokens, bigramsCounter)

    # Now use the test file to calculate probabilities
    # Read the test file and calculate unigram and 
    # bigram model probabilities for each sentence.
    __format = lambda p: "undefined" if p == 0 else "{:.4f}".format(p)

    # Test the Ngram model on test file and calculate the 
    # sentence prediction probability
    with open(testFile, 'r') as testFile:
        for line in testFile:
            sentence = line.lower().strip()
            uniProb = GetUnigramSentenceProbability(sentence)
            biProb = GetBigramSentenceProbability(sentence)
            biSmoothProb = GetSmoothedSentenceProbability(sentence)

            print("S =", line.strip())
            print("Unigrams: logprob(S) =", __format(uniProb))
            print("Bigrams: logprob(S) =", __format(biProb))
            print("Smoothed Bigrams: logprob(S) =", __format(biSmoothProb))
            print()

if __name__ == '__main__':
    main(sys.argv[1:])