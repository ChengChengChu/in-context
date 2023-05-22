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
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

def log_dict_sort(e):
    return e["Average_Reward"]

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

    t = time.time()
    t1 = time.localtime(t)
    t2 = time.strftime('%Y_%m_%d_%H_%M_%S',t1)
    log_path = os.path.join(args.save_path, f"{t2}.csv")

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

    dict_format = {"Turn": None, "Prompt": None, "Demos": None, "Dialogue+Reward": [], "Average_Reward": None}
    log_dict = [dict(dict_format) for _ in range(args.sample_num + 1)]

    # analyzer = SentimentIntensityAnalyzer()

    log_dict[-1]["Prompt"] = ""
    
    generate_proposal(args.proposal_template_path, args.sample_num, args.proposal_temperature, log_dict, args)
    # prompts = ["prompt" for _ in range(args.sample_num)]
    
    for i in tqdm(range(args.resample_turn_num + 1), desc="Running through turns"):
        for j in tqdm(range(len(log_dict)), desc="Running through all prompts"):
            total_reward = 0
            for k in tqdm(range(len(prefix)), desc="Running through all prefix"):
                if args.fix_speakerA:
                    dialogue = make_dialogue_fix_A(log_dict[j]["Prompt"], Bot, args, prefix[k])
                else:
                    dialogue = make_dialogue(log_dict[j]["Prompt"], args.multi_turn_num, Bot, Interlocutor, args, prefix[k])

                if args.reward == 'comfort':
                    reward = comfort_reward(dialogue)
                total_reward += reward
                log_dict[j]["Dialogue+Reward"].append({"Reward": reward, "Dialogue": dialogue_to_readable(dialogue)})
            averge_reward = total_reward / len(prefix)
            log_dict[j]["Average_Reward"] = averge_reward   

        # prompt_reward = comfort_reward()
        log_dict.sort(reverse = True, key=log_dict_sort)
        pd.DataFrame(log_dict).to_csv(log_path, mode = "a", index=False, header=False)
        # with open(log_path, 'a') as f:
        #     f.write(f"Turn{i}:\n\n")
        #     for prompt_reward in prompt_rewards:
        #         f.write(f"averge_reward: {prompt_reward['averge_reward']}\n")
        #         f.write(f"prompt: {prompt_reward['prompt']}\n")
        #         f.write("dialogues:\n")
        #         for k in range(len(prompt_reward['dialogues'])):
        #             f.write(f"reward:{prompt_reward['rewards'][k]}\n")
        #             for s in range(len(prompt_reward['dialogues'][k]['A'])):                      
        #                 f.write(f"SpeakerA: {prompt_reward['dialogues'][k]['A'][s]}\n")
        #                 f.write(f"SpeakerB: {prompt_reward['dialogues'][k]['B'][s]}\n")
        #             f.write('\n')
        log_dict = log_dict[:args.top_k_prompts]
        # prompt_rewards = prompt_rewards[:args.top_k_prompts]
        old_prompts = [l["Prompt"] for l in log_dict]
        new_prompts = []

        if i != args.resample_turn_num:
            for j in tqdm(range(len(old_prompts)), desc="Resampling"):
                for _ in range(args.resample_num):
                    messages=[{"role": "user", "content": f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {old_prompts[j]}\nOutput:<COMPLETE>"}]
                    new_prompts.append(openai_chat_response(messages, args.resample_temperature))
        log_dict = []
        for p in new_prompts:
            new_dict = dict(dict_format)
            new_dict["Prompt"] = p
            new_dict["Turn"] = i + 1
            log_dict.append(new_dict)
    # for prompt_reward in prompt_rewards:
    #   print(f"reward:{prompt_reward[0]}")
    #   print(f"prompt:{prompt_reward[1]}")

    #   dialogue = prompt_reward[2]
    #   print(f"prefix:{dialogue[0]}")
    #   for i in range(1, len(dialogue)):
    #     if i % 2 == 0:
    #       print(f"interlocutor:{dialogue[i]}")
    #     else:
    #       print(f"bot:{dialogue[i]}")
    #   print("")
    #   #sort prompt_reward
      #pick topK to resample
    
    
    

    


    # df = pd.read_csv(args.prompt_path)
    # sentences = df['prompt'].tolist()
    # result = []
    
    # for sens in tqdm(sentences) :
    #     if args.bot == 'blenderbot' :
    #         if len(sens) >= 128 : sens = sens[:128]
    #     score, re_sen, re_res = bias_reward([sens], Bot, analyzer)
    #     tmp = [score[0], re_sen[0][0], re_sen[0][1], re_res[0][0], re_res[0][1]]
    #     result.append(tmp)
    
    # df = pd.DataFrame(result, columns=['score', 'send_1', 'send_2', 'response_1', 'response_2'])
    # if not os.path.exists('result') :
    #     os.mkdir('result')
    # df.to_csv('result/' + args.save_path)
    
        


def set_arguments(parser):
    parser.add_argument("--proposal_template_path", type=str, default="template/general.txt", help="Path to the template proposal")
    parser.add_argument("--reward", type=str, default="comfort")
    parser.add_argument("--prefix_data_path", type=str, default="data/delta_test.txt")
    parser.add_argument("--prefix_sample_num", type=int, default=1)
    parser.add_argument("--fix_speakerA", action="store_true")
    parser.add_argument("--demo_data_path", type=str, default=None)
    parser.add_argument("--demo_num", type=int, default=5)
    parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--top_k_prompts", type=int, default=2)
    parser.add_argument("--resample_num", type=int, default=10)
    parser.add_argument("--resample_turn_num", type=int, default=5)
    parser.add_argument("--proposal_temperature", type=float, default=1.0)
    parser.add_argument("--resample_temperature", type=float, default=1.0)
    parser.add_argument("--multi_turn_num", type=int, default=1) 
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