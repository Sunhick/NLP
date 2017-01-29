#!/usr/bin/python3

import re
import os
import sys

class WordCount(object):
    def __init__(self, filename):
        if os.path.isfile(filename):
             self.filename = filename
             self.__wc()
        else:
            raise Exception("File \"{0}\", doesn't exists!".format(self.filename))

    def __wc(self):
        with open(self.filename) as file:
            # Fix me: Should i ignore apostrophe, Abbrevations, Periods, Numbers(45.45)
            data = file.read().lower().replace("\'", "").replace(".", "") #.replace("-", "")
            self.words = re.findall('\w+', data)

            # Paragraphs are separated by atleast two new lines
            self.paragraphs = re.findall('\n{2,}', data)

            # Sentences are separated by a newline
            self.sentences = re.findall('\n', data)

    @property
    def Paragraphs(self):
        return len(self.paragraphs)

    @property
    def Sentences(self):
        return len(self.sentences)

    @property
    def Words(self):
        return len(self.words)

def main(argv):
    filename = argv[0]
    wc = WordCount(filename)
    print("Paragraphs:", wc.Paragraphs)
    print("Sentences:", wc.Sentences)
    print("Words:", wc.Words)

if __name__ == "__main__":
    main(sys.argv[1:])