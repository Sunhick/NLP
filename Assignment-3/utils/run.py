#!/usr/bin/python3

import os

def main():
    baseTestFile = "data/test_{0}.txt"
    baseTrainFile = "data/train_{0}.txt"

    for i in range(1, 30):
        testFile = baseTestFile.format(i)
        trainFile = baseTrainFile.format(i)

        os.system("{cmd} {pythonFile} {train_File} {test_File}".
            format(cmd="python3", pythonFile="POSTagger.py", 
                train_File = trainFile, test_File = testFile))

if __name__ == "__main__":
    main()