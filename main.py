import numpy as np
import torch
import random
import argparse
import importlib
from tqdm import tqdm
import shutil
from pathlib import Path
from argparse import ArgumentParser
import os
from argparse import Namespace
import warnings
from os.path import join
import json
from utils import *
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    Bot = bot(args)

    df = pd.read_csv(args.prompt_path)
    sentences = df['prompt'].tolist()
    
    for sens in sentences :
        sen_1, sen_2, gen = replace_sentence(sens)

        print(sen_1, sen_2, gen)
        import pdb
        pdb.set_trace()
    
    
   

def set_arguments(parser):
    parser.add_argument("--prompt_path", type=str, default="") 
    parser.add_argument("--bot", type=str, default="example")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="") # save path
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=.9)
    

    args = parser.parse_args()

    return args

def fix_seed(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    return

if __name__ == "__main__":
    main()    