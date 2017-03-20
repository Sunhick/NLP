#!/usr/bin/python3

"""
POS tagging using HMM (Viterbi algorithm)
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"
__version__ = "0.1"
import os
import sys
import random

# not required for now. as we are not splitting contractions ourself.
# but requried when testing with sentences.
# from nltk.tokenize import *

# to create bigram partial class from ngram
# from functools import partial
# Bigrams = partial(Ngrams, n = 2)

from abc import ABCMeta, abstractmethod

# for calculating log probabilities
from math import log10
from copy import deepcopy
# for getting the max in tuple based on key
from operator import itemgetter
# for getting the max of ProbEntry 
from operator import attrgetter
from collections import defaultdict
from collections import namedtuple

# progress bar
from status import printProgressBar

class POSError(Exception):
    """
    Defines the POS tagger application error
    """
    pass

class Constants(object):
    kSENTENCE_BEGIN = "<s>"
    kSENTENCE_END = "</s>"

class WordTag(namedtuple('WordTag', ['word', 'tag'], verbose=False)):
    """
    Represents the word tag pair. Inherits from namedtuple so that i can 
    unpack the class in word, tag pair easily in the iterations.
    """
    word = tag = None

    def __init__(self, word, tag):
        # TODO: shoud i ignore the case of word?
        self.word = word.lower()
        self.tag = tag

    def IsLastWord(self):
        return self.word == Constants.kSENTENCE_END

    def IsFirstWord(self):
        return self.word == Constants.kSENTENCE_BEGIN

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

    @property
    def Sentence(self):
        words = [wt.word if (not wt.IsFirstWord() and not wt.IsLastWord()) else ""
                     for wt in self.words]
        return " ".join(words).strip()

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

class POSFile(object):
    """
    Abstraction over words-tags, lines as a POSfile.
    """
    lines = []

    def __init__(self, filename):
        # os.path.exists is a false positive if we are looking for file.
        # os.path.exists only checks if there's an inode entry in dir and doesn't
        # check the type of file in that directory.
        if not os.path.isfile(filename):
            raise POSError("{0} is invalid file".format(filename))

        self.__read(filename)

    def __read(self, filename):
        sentence = Line()
        sentence.AddWordTag(Constants.kSENTENCE_BEGIN, Constants.kSENTENCE_BEGIN)
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip():
                    # end of sentence
                    self.lines.append(sentence)

                    # create a new line holder
                    sentence = Line()
                    # add the word begin marker
                    sentence.AddWordTag(Constants.kSENTENCE_BEGIN, Constants.kSENTENCE_BEGIN)
                    continue

                word, tag = line.split()

                # TODO: Should i ignore the periods?
                # Marks the last word in the sentence
                if word == "." and tag == ".":
                    # add the word end marker
                    sentence.AddWordTag(Constants.kSENTENCE_END, Constants.kSENTENCE_END)
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
    Generates the ngrams from the list of words.
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
    """
    Represents the bigrams tokens. This is a
    vanilla class that inherits from Ngrams.
    """
    def __init__(self, words):
        super(Bigrams, self).__init__(words, n = 2)

class ProbEntry(object):
    """
    Represents path probability entry in viterbi matrix 
    """

    # store log probability for easier calculations, and also
    # we don't lose the floating point precision.
    probability = float(0)
    backpointer = None
    tag = None
    word = None

    def __init__(self, probability=0.0, tag=None, backpointer=None):
        self.probability = probability
        self.backpointer = backpointer
        self.tag = tag

    def __str__(self):
        backptr = id(self.backpointer) if self.backpointer else None
        return "Prob={0} id={2} tag={3} word={4} BackPtr={1}".     \
            format(self.probability, backptr, id(self), self.tag, self.word)

class Decoder(object):
    """
    Decoder interface 
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, tagger, sentence):
        raise NotImplementedError("Should implement this callable method")

class Viterbi(Decoder):
    """
    Stateless viterbi algorithm to decode the sequence using bigram model.
    """
    end = start = None

    def __init__(self):
        self.start = "start"
        self.end = "end"

    def __backTrack(self, viterbi, T):
        """
        Return the tag sequence by following back pointers to states
        back in time from viterbi.backpointers
        """
        tagSequence = []
        pointer = viterbi[T][self.end][self.end].backpointer
        
        # traverse the back pointers
        while (pointer):
            # print("TAG=%s WORD=%s" % (pointer.tag, pointer.word))
            tagSequence.append(pointer.tag)
            pointer = pointer.backpointer

        # reverse the tag sequence as they are 
        # traced from back to front
        tagSequence.reverse()
        return tagSequence

    def __call__(self, tagger, sentence):
        """
        callable method on instance. Takes a hmm tagger instance and sentence.
        hmm tagger instance provides the tag transition probability and 
        likelihood probability requried for calculating the tag sequence.
        """
        tagSequence = []
        # implement viterbi algorithm here
        N = tagger.V
        tokens = sentence.split()
        T = len(tokens)
        # viterbi = [[ProbEntry() for j in range(T)] for i in range(N+2)]

        viterbi = [defaultdict(lambda: defaultdict(ProbEntry)) for i in range(T+2)]

        # viterbi[i] makes sures entries are unique even if 
        # a given (word, tag) occurs multiple times in the sentence.
        # Without viterbi[i], there will be circular back pointers, thereby 
        # not leading to a infinite set of tags.
        viterbi[0][self.start][self.start] = None

        # initialization step
        for state in tagger.tagset:
            viterbi[1][state][tokens[0]].probability = log10(1)                             \
                + tagger.GetTagTransitionProbability(Constants.kSENTENCE_BEGIN, state)      \
                + tagger.GetLikelihoodProbability(state, tokens[0])
            viterbi[1][state][tokens[0]].tag = state
            viterbi[1][state][tokens[0]].word = tokens[0]
            viterbi[1][state][tokens[0]].backpointer = viterbi[0][self.start][self.start]

        # recursion step
        for time in range(1, T):
            # print("w =", tokens[time])
            for state in tagger.tagset:
                # tuple of (prob. entry and prob. value)
                prbs = [
                    (viterbi[time][sp][tokens[time-1]], 
                        viterbi[time][sp][tokens[time-1]].probability                   \
                        + tagger.GetTagTransitionProbability(sp, state)                 \
                        + tagger.GetLikelihoodProbability(state, tokens[time]))
                    for sp in tagger.tagset 
                    ]

                backptr, prob = max(prbs, key=itemgetter(1))
                viterbi[time+1][state][tokens[time]].probability = prob
                viterbi[time+1][state][tokens[time]].tag = state
                viterbi[time+1][state][tokens[time]].word = tokens[time]
                viterbi[time+1][state][tokens[time]].backpointer = backptr

            # maxd = max([viterbi[time+1][state][tokens[time]] for state in tagger.tagset],
            #             key = attrgetter("probability"))
            # print(maxd)

        # termination step
        final = max([viterbi[T][s][tokens[T-1]] for s in tagger.tagset],      \
                        key=attrgetter("probability"))
        viterbi[T][self.end][self.end].backpointer = final
        # print("final=", final)

        # return the backtrace path by following back pointers to states
        # back in time from viterbi.backpointers
        tagSequence = self.__backTrack(viterbi, T)
        return tagSequence

class FastViterbi(Decoder):
    """
    Stateless faster viterbi algorithm to decode the sequence using bigram model.
    """
    end = start = None

    def __init__(self):
        self.start = "start"
        self.end = "end"

    def __call__(self, tagger, sentence):
        """
        callable method on instance. Takes a hmm tagger instance and sentence.
        hmm tagger instance provides the tag transition probability and 
        likelihood probability requried for calculating the tag sequence.
        """
        raise POSError("Use regular Viterbi! This has to be implemented" 
            + " correctly and faster than regular viterbi.")

        tagSequence = []
        # implement viterbi algorithm here
        N = tagger.V
        tokens = sentence.split()
        T = len(tokens)

        viterbi = [defaultdict(lambda: defaultdict(ProbEntry)) for i in range(2)]

        # viterbi[i] makes sures entries are unique even if 
        # a given (word, tag) occurs multiple times in the sentence.
        # Without viterbi[i], there will be circular back pointers, thereby 
        # not leading to a infinite set of tags.
        level = 0
        viterbi[level][self.start][self.start] = None

        level = 1
        # initialization step
        for state in tagger.tagset:
            viterbi[level][state][tokens[0]].probability = log10(1)                             \
                + tagger.GetTagTransitionProbability(Constants.kSENTENCE_BEGIN, state)          \
                + tagger.GetLikelihoodProbability(state, tokens[0])
            viterbi[level][state][tokens[0]].tag = state
            viterbi[level][state][tokens[0]].word = tokens[0]
            viterbi[level][state][tokens[0]].backpointer = viterbi[0][self.start][self.start]

        maxd = max([viterbi[level][state][tokens[0]] for state in tagger.tagset],
                        key = attrgetter("probability"))
        tagSequence.append(maxd.tag)
        # recursion step
        for time in range(1, T):
            max_entry = None
            for state in tagger.tagset:
                # tuple of (prob. entry and prob. value)
                prbs = [
                    (viterbi[level][sp][tokens[time-1]], 
                        viterbi[level][sp][tokens[time-1]].probability                      \
                        + tagger.GetTagTransitionProbability(sp, state)                     \
                        + tagger.GetLikelihoodProbability(state, tokens[time]))
                    for sp in tagger.tagset 
                    ]

                level = (level+1)%2
                # reset the previous level.
                viterbi[level] = defaultdict(lambda: defaultdict(ProbEntry))

                backptr, prob = max(prbs, key=itemgetter(1))
                viterbi[level][state][tokens[time]].probability = prob
                viterbi[level][state][tokens[time]].tag = state
                viterbi[level][state][tokens[time]].word = tokens[time]
                viterbi[level][state][tokens[time]].backpointer = backptr
                
                entry = viterbi[level][state][tokens[time]]
                if not max_entry:
                    max_entry = entry
                else:
                    max_entry = entry if entry.probability > max_entry.probability else max_entry
                # max_entry = entry if max_entry and entry.probability > max_entry.probability else max_entry

            tagSequence.append(max_entry.tag)

        # termination step
        level = (level+1)%2
        final = max([viterbi[level][s][tokens[T-1]] for s in tagger.tagset],      \
                        key=attrgetter("probability"))
        viterbi[level][self.end][self.end].backpointer = final
        tagSequence.append(final.tag)
        return tagSequence

class HMMTagger(object):
    """
    POS tagger using HMM. Each word may have more tags assosicated with it.
    """
    tagTransitions = likelihood = None
    V = 0       # vocabulary size. Total count of tags
    tagset = set() # vocabulary set/ different POS tags
    k = 0.0  # maybe i have to fine tune this to get better accuracy.
    __decoder = None

    def __init__(self, k = 0.0001, decoder = Viterbi()):
        """
        Initialize the varibles. decoder is paramterized and default decoder is viterbi.
        Default viterbi uses bigram model sequence. If you want use your own decoder, then 
        define it and pass it the HMM tagger.

        Defining a decoder: It should be callable on object instance i.e impement 
        __call__() method. Signature : def __call__(self, hmm_instance, sentence)
        """
        if not issubclass(type(decoder), Decoder):
            raise POSError("{0} doesn't implement interface {1}".format(decoder, Decoder))

        self.tagTransitions = defaultdict(lambda: defaultdict(float))
        self.likelihood = defaultdict(lambda: defaultdict(float))
        self.k = k
        self.__decoder = decoder

    def Train(self, trainData):
        """
        Train the HMM using train data. i.e calculate the 
        likelihood probabilities and tag transition probabilities.
        """
        for line in trainData:
            # update the likelihood probabilities
            for wordtag in line:
                if not wordtag.IsFirstWord() and not wordtag.IsLastWord():
                    word, tag = wordtag
                    # unpack word and tag. I can do this becusae of namedtuple
                    self.likelihood[tag][word] += 1
                    self.tagset.add(tag)

            words = line.words
            # update the tag transition probabilties
            for first, second in Bigrams(words).Ngrams:
                _, fromTag = first
                _, toTag = second
                self.tagTransitions[fromTag][toTag] += 1

        # Normalize probablities
        self.__normalize()

    def __normalize(self):
        """
        Normalize the tag transition table and likelihood proabibility table.
        For easier and faster look up.
        """
        # -1 because of <s>
        # self.tagset = set(self.tagTransitions).remove(Constants.kSENTENCE_BEGIN)
        self.V = len(self.tagset)

        # If i normalize the tag transition table, 
        # I can directly use it and no need for 
        # below two methods.
        # TODO: To be implemented.

    def GetTagTransitionProbability(self, fromTag, toTag):
        """
                      C(X, Y) + k
        P(Y | X) =   ______________
                        C(X) + Vk

        Use add-k with k = 0.0001 ? (assignment-2 value)
        """
        # return log10(self.tagTransitions[fromTag][toTag]+.0000001)
        prob = 0.0
        cxy = self.tagTransitions[fromTag][toTag]
        cx = sum(self.tagTransitions[fromTag].values())
        prob = (cxy + self.k) / (cx + (self.k * self.V))
        return float(prob)

    def GetLikelihoodProbability(self, tag, word):
        """
                            C(tag, word) + k
        P(word | tag) =    ___________________
                              C(tag) + Vk

        Use add-k with k = 0.0001 ? (assignment-2 value)
        """
        # return log10(self.likelihood[tag][word]+.00000001)
        prob = 0.0
        ctagword = self.likelihood[tag][word]
        ctag = sum(self.likelihood[tag].values())
        prob = (ctagword + self.k) / (ctag + (self.k * self.V))
        return float(prob)

    def Log10TransProbability(self, fromTag, toTag):
        try:
            return log10(self.GetTagTransitionProbability(fromTag, toTag))
        except ValueError:
            # If there's any math domain error. Just return probability as 0.
            return float(0)

    def Log10LikelihoodProbability(self, tag, word):
        try:
            return log10(self.GetLikelihoodProbability(tag,word))
        except ValueError:
            # If there's any math domain error. Just return probability as 0.
            return float(0)

    def Decode(self, sentence):
        return self.__decoder(self, sentence)

def main(args):
    filename = args[0]
    # train(filename)
    data = POSFile(filename)

    # split data as 80:20 for train and test 
    train, test = data.RandomSplit(80)
    tagger = HMMTagger(k = 0.00001, decoder = Viterbi())
    tagger.Train(train)

    formatter = lambda word, tag: "{0}\t{1}\n".format(word, tag)
    endOfSentence = ".\t.{newline}{newline}".format(newline=os.linesep)

    current = 0
    total = len(test)
    print("\nDecoding the tag sequence for test data...\n")
    with open("berp-key.txt", "w") as goldFile,         \
         open("berp-out.txt", "w") as outFile:
        for line in test:
            sentence = line.Sentence
            # print("S =", sentence)
            tagSequence = tagger.Decode(sentence)

            for wt in line:
                if not wt.IsFirstWord() and not wt.IsLastWord():
                    w, t = wt
                    goldFile.write(formatter(w, t))

            goldFile.write(endOfSentence)

            for w, t in zip(sentence.split(), tagSequence):
                outFile.write(formatter(w, t))
            outFile.write(endOfSentence)

            current += 1
            printProgressBar(current, total, prefix="Progress:", suffix="Complete", length=50)

if __name__ == "__main__":
    main(sys.argv[1:])