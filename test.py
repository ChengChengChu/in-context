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
from generate import generate_proposal
from dialogue import make_dialogue, make_dialogue_fix_A
import time
from openai_generate_response import openai_chat_response
import pandas as pd
import pickle
import copy
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

def dialogue_to_readable(dialogue):
    # dialogue = {'A':[], 'B':[]}
    s = ""
    for i in range(len(dialogue['A'])):
        s += f"SpeakerA: {dialogue['A'][i]}\n"
        s += f"SpeakerB: {dialogue['B'][i]}\n"
    return s


def main():

    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)

    prompts = []
    with open(args.prompt_file) as f:
        table = f.readlines()
    for t in table:
        prompt = t.split('[SEP]')
        for i in range(len(prompt)):
            prompt[i] = prompt[i].strip()
        prompts.append(prompt)

    os.makedirs(f"{args.save_path}/test", exist_ok=True)

    t = time.time()
    t1 = time.localtime(t)
    t2 = time.strftime('%Y_%m_%d_%H_%M_%S',t1)
    log_path_csv = os.path.join(args.save_path, "test", f"test_{t2}.csv")
    log_path_pkl = os.path.join(args.save_path, "test", f"test_{t2}.pkl")

    openai.api_key = args.openai_api
    if args.openai_org is not None:
        openai.organization = args.openai_org
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Import model of bot and interlocutor from bots
    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    if args.bot == "gpt3":
        Bot = bot(args)
    else:
        Bot = bot(args.bot_model_path, args)
    # Bot = bot(args.bot_model_path, args)
    interlocutor = importlib.import_module(".module", f"bots.{args.interlocutor}").bot

    if args.interlocutor == "gpt3":
        Interlocutor = interlocutor(args)
    else:
        Interlocutor = interlocutor(args.interlocutor_model_path, args)

    # Interlocutor = interlocutor(args.interlocutor_model_path, args)

    if args.prefix_data_path is not None:
        prefix = []
        with open(args.prefix_data_path, 'r') as f:
            table = f.readlines()
        d = {'A':[], 'B':[]}
        for t in table:
            if t == "====================================================================================================\n":
                prefix.append(d)
                d = {'A':[], 'B':[]}
            else:
                if t.find("Speaker A:") >= 0:
                    d['A'].append(t[11:].strip().replace("\"", ""))
                elif t.find("Speaker B:") >= 0:
                    d['B'].append(t[11:].strip().replace("\"", ""))

    prefix = random.sample(prefix, args.prefix_sample_num)

    dict_format = {"Prompt": None, "Dialogues": [], "Dialogues_split": []}
    log_dict = [copy.deepcopy(dict_format) for _ in range(len(prompts))]

    for i in tqdm(range(len(log_dict))):
        log_dict[i]['Prompt'] = prompts[i]
        for j in tqdm(range(args.prefix_sample_num)):
            dialogue = dict()
            dialogue = make_dialogue(prompts[i], args.multi_turn_num, Bot, Interlocutor, args, prefix[j])
            log_dict[i]["Dialogues"].append(dialogue_to_readable(dialogue))
            log_dict[i]["Dialogues_split"].append(dialogue)
    pd.DataFrame(log_dict).to_csv(log_path_csv, index=False)
    with open(log_path_pkl, 'wb') as f:
        pickle.dump(log_dict, f)
    
    
    
        


def set_arguments(parser):
    parser.add_argument("--prompt_file", type=str)
    # parser.add_argument("--proposal_template_path", type=str, default="template/general.txt", help="Path to the template proposal")
    # parser.add_argument("--reward", type=str, default="comfort")
    parser.add_argument("--prefix_data_path", type=str, default="data/delta_test.txt")
    parser.add_argument("--prefix_sample_num", type=int, default=1)
    # parser.add_argument("--fix_speakerA", action="store_true")
    # parser.add_argument("--demo_data_path", type=str, default=None)
    # parser.add_argument("--demo_num", type=int, default=5)
    # parser.add_argument("--sample_num", type=int, default=1000)
    # parser.add_argument("--top_k_prompts", type=int, default=2)
    # parser.add_argument("--resample_num", type=int, default=10)
    # parser.add_argument("--resample_turn_num", type=int, default=5)
    # parser.add_argument("--proposal_temperature", type=float, default=1.0)
    # parser.add_argument("--resample_temperature", type=float, default=1.0)
    parser.add_argument("--multi_turn_num", type=int, default=3) 
    parser.add_argument("--bot", type=str, default="gpt3")
    parser.add_argument("--bot_model_path", type=str, default=None)
    parser.add_argument("--interlocutor", type=str, default="gpt3")
    parser.add_argument("--interlocutor_model_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="results/") # save path
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=.9)
    parser.add_argument("--openai_api", type=str)
    parser.add_argument("--openai_org", type=str, default=None)
    

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