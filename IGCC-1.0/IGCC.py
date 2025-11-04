#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter
from argparse import ArgumentParser

from igcc._igcc import IGCC


def main():



    # Provide arguments.
    parser = ArgumentParser(usage='%(prog)s -i smiles.txt -p parameters.py')
    parser.add_argument('-i', '--smi', default='smiles.txt', help='Input SMILES file')
    parser.add_argument('-p', '--para', default='parameters.py', help='Input parameters file')
    
    # Obtain arguments.
    args = parser.parse_args()
    input_file = args.smi
    para_file = args.para

    # run procedure.
    work = IGCC(input_file='smiles.txt', para_file='parameters.ini')
    work.execute()
    input("Please input any key to exit!")



if __name__ == '__main__':
    main()
