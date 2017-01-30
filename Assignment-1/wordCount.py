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

class WordCount(object):
    def __init__(self, filename):
        if os.path.isfile(filename):
             self.filename = filename
             self.__wc()
        else:
            raise Exception("File \"{0}\", doesn't exists!".format(filename))

    def __wc(self):
        with open(self.filename) as file:
            # Fix me: Should i ignore apostrophe, Abbrevations, Periods, Numbers(45.45)
            data = file.read().lower().replace("\'", "").replace(".", "")\
                    .replace("-", "")
            # print(data)
            self.words = len(re.findall(r'\w+', data))
            # print(self.words)

            # Paragraphs are separated by atleast two new lines
            self.paragraphs = len(re.findall(r'\n{2,}', data)) + 1

            # Sentences are separated by a newline
            self.sentences = len(re.findall(r'\n', data))

    @property
    def Paragraphs(self):
        return self.paragraphs

    @property
    def Sentences(self):
        return self.sentences

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