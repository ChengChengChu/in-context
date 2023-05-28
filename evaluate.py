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
from epi_reward import get_reward
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

    os.makedirs(f"{args.save_path}/evaluate", exist_ok=True)

    t = time.time()
    t1 = time.localtime(t)
    t2 = time.strftime('%Y_%m_%d_%H_%M_%S',t1)
    log_path_csv = os.path.join(args.save_path, "evaluate", f"evaluate_{t2}.csv")
    log_path_pkl = os.path.join(args.save_path, "evaluate", f"evaluate_{t2}.pkl")

    openai.api_key = args.openai_api
    if args.openai_org is not None:
        openai.organization = args.openai_org

    with open(args.dialogue_file, "rb") as f:
        # {'Prompt': str, 'Dialogues': [str, str, ...], 'Dialogues_split': [{'A': [str, str, ...], 'B': [str, str, ...]}, ...]}
        dialogue_dict = pickle.load(f)

    dict_format = {"Prompt": None, "Dialogue": None, "Bot_length": None, "Inter_length": None, "epi_ip": None, "epi_ex": None, "epi_er": None}
    log_dict = []

    for i in tqdm(range(len(dialogue_dict))):
        for j in range(len(dialogue_dict[i]["Dialogues"])):
            new_dict = copy.deepcopy(dict_format)
            new_dict["Prompt"] = dialogue_dict[i]["Prompt"]
            dialogue = dialogue_dict[i]["Dialogues"][j]
            dialogue_split = dialogue_dict[i]["Dialogues_split"][j]
            new_dict["Dialogue"] = dialogue
            new_dict["Bot_length"] = (sum(len(x.split()) for x in dialogue_split["B"][:]))
            new_dict["Inter_length"] = (sum(len(x.split()) for x in dialogue_split["A"][1:]))
            # "IP_score": ips, "EX_score": exs, "ER_score": ers
            epi_reward = get_reward(dialogue)
            new_dict["epi_ip"] = epi_reward["IP_score"]
            new_dict["epi_ex"] = epi_reward["EX_score"]
            new_dict["epi_er"] = epi_reward["ER_score"]
            log_dict.append(new_dict)
    pd.DataFrame(log_dict).to_csv(log_path_csv, index=False)
    with open(log_path_pkl, 'wb') as f:
        pickle.dump(log_dict, f)
    
        


def set_arguments(parser):
    parser.add_argument("--dialogue_file", type=str)
    # parser.add_argument("--proposal_template_path", type=str, default="template/general.txt", help="Path to the template proposal")
    # parser.add_argument("--reward", type=str, default="comfort")
    # parser.add_argument("--prefix_data_path", type=str, default="data/delta_test.txt")
    # parser.add_argument("--prefix_sample_num", type=int, default=1)
    # parser.add_argument("--fix_speakerA", action="store_true")
    # parser.add_argument("--demo_data_path", type=str, default=None)
    # parser.add_argument("--demo_num", type=int, default=5)
    # parser.add_argument("--sample_num", type=int, default=1000)
    # parser.add_argument("--top_k_prompts", type=int, default=2)
    # parser.add_argument("--resample_num", type=int, default=10)
    # parser.add_argument("--resample_turn_num", type=int, default=5)
    # parser.add_argument("--proposal_temperature", type=float, default=1.0)
    # parser.add_argument("--resample_temperature", type=float, default=1.0)
    # parser.add_argument("--multi_turn_num", type=int, default=3) 
    # parser.add_argument("--bot", type=str, default="gpt3")
    # parser.add_argument("--bot_model_path", type=str, default=None)
    # parser.add_argument("--interlocutor", type=str, default="gpt3")
    # parser.add_argument("--interlocutor_model_path", type=str, default=None)
    # parser.add_argument("--exp_name", type=str, default="")
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