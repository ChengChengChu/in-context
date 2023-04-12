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
import openai
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # replace yaml default argument to argparser arg. 

    openai.organization = "org-9VP7zbu5OprKdttIEI0m2wqX"
    openai.api_key = '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'

    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    Bot = bot(args)


    # analyzer = SentimentIntensityAnalyzer()

    df = pd.read_csv(args.prompt_path)
    sentences = df['prompt'].tolist()
    result = []
    
    for sens in tqdm(sentences) :
        if args.bot == 'blenderbot' :
            if len(sens) >= 128 : sens = sens[:128]
        score, re_sen, re_res = bias_reward([sens], Bot, analyzer)
        tmp = [score[0], re_sen[0][0], re_sen[0][1], re_res[0][0], re_res[0][1]]
        result.append(tmp)
    
    df = pd.DataFrame(result, columns=['score', 'send_1', 'send_2', 'response_1', 'response_2'])
    if not os.path.exists('result') :
        os.mkdir('result')
    df.to_csv('result/' + args.save_path)
    
        


def set_arguments(parser):
    parser.add_argument("--template_path", type=str, default="")
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--top_k_prompts", type=int, default=10)
    parser.add_argument("--resample_num", type=int, default=10)
    parser.add_argument("--reample_turn_num", type=int, default=5)
    parser.add_argument("--proposal_temperature", type=float, default=1.0)
    parser.add_argument("--resample_temperature", type=float, default=1.0)
    parser.add_argument("--multi_turn_num", type=int, dafault=1) 
    parser.add_argument("--bot", type=str, default="gpt3")
    parser.add_argument("--interlocutor", type=str, default="gpt3")
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