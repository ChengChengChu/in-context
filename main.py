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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    Bot = bot(args)

    analyzer = SentimentIntensityAnalyzer()

    df = pd.read_csv(args.prompt_path)
    sentences = df['prompt'].tolist()
    result = []
    o = 0
    
    for sens in tqdm(sentences) :
        score, re_sen, re_res = bias_reward([sens], Bot, analyzer)
        tmp = [score[0], re_sen[0][0], re_sen[0][1], re_res[0][0], re_res[0][1]]
        result.append(tmp)
        o += 1
        if o >= 2 : break
    
    df = pd.DataFrame(result, columns=['score', 'send_1', 'send_2', 'response_1', 'response_2'])
    if not os.path.exists('result') :
        os.mkdir('result')
    df.to_csv('result/' + args.save_path)
    
        


def set_arguments(parser):
    parser.add_argument("--prompt_path", type=str, default="") 
    parser.add_argument("--bot", type=str, default="example")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="result.csv") # save path
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