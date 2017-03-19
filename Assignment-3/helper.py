#!/usr/bin/python3

"""
helper file for POSTagger
"""

__author__ = "Sunil"
__email__ = "suba5417@colorado.edu"

import os
import sys
import math

def generate(filename):
    s = set()
    with open(filename, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            word, tag = line.split()
            if word == "." and tag == ".":
                continue
            s.add(tag)
    return s