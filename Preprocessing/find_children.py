from pmaw import PushshiftAPI
import datetime as dt 
import pandas as pd
import argparse as arg
import os, sys
import numpy as np
import logging

from api_find_subs import get_crosspost_child



def main():
    df = pd.read_pickle('../../Files/test_0613_raw2.pickle')
    depth_lim = 2
    outfile = 'child_out_0613'
    get_crosspost_child(df, outfile, depth_lim)

if __name__ == '__main__':
    main()
