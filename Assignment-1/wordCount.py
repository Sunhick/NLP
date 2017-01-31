#!/usr/bin/python3

"""
Word count: 

Counts the total number of words, lines and paragraphs in a given file.
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"

import re
import os
import sys
import functools

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

class WordCount(object):
    def __init__(self, filename):
        if os.path.isfile(filename):
             self.filename = filename
             self.__wc()
        else:
            raise Exception("File \"{0}\", doesn't exists!".format(filename))

    def __removeDottedAbbr(self, match):
        return match.group().replace(".", "")

    def replaceApostophe(self, text):
        # collapse words with apostrophe.
        return text.replace("\'", "")

    def replaceHyphen(self, text):
        #  collapse words with hyphens.
        return text.replace("-", "")

    def removeDottedAbbr(self, text):
        # Collapse abbrevations like U.S.A => USA, m.p.h => mph etc
        removeDottedAbbr = re.compile(r'((?:[a-zA-Z]\.){2,})')
        return removeDottedAbbr.sub(self.__removeDottedAbbr, text)

    def __wc(self):
        # Fix me: Should i ignore apostrophe, Abbrevations, Periods, Numbers(45.45)
        # Pre-process the text 
        self.textPreProcessPipeline = compose(self.replaceApostophe, self.replaceHyphen, self.removeDottedAbbr)
        
        with open(self.filename) as file:
            data = self.textPreProcessPipeline(file.read())

            self.words = len(re.findall(r"\w+", data))
            # print(self.words)

            # Paragraphs are separated by atleast two new lines
            self.paragraphs = len(re.findall(r"\n{2,}", data)) + 1

            # lines are separated by a newline
            self.lines = len(re.findall(r"\n", data))

            # sentences 
            self.sentences = len(re.findall(r"[^\s](\.|\!|\?)(?!\w)", data))

    @property
    def Paragraphs(self):
        return self.paragraphs

    @property
    def Sentences(self):
        return 100

    @property
    def Lines(self):
        return self.lines

    @property
    def Words(self):
        return self.words

def main(cmdline):
    filename = cmdline[0]
    try:
        wc = WordCount(filename)
        print("Paragraphs:", wc.Paragraphs)
        print("Sentences:", wc.Sentences)
        print("Words:", wc.Words)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main(sys.argv[1:])