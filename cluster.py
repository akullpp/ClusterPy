#!/usr/bin/env python

from bhmm import BHMM
from utility import parse_args

if __name__ == "__main__":
    args = parse_args()
    bhmm = BHMM(args)
    bhmm.run()