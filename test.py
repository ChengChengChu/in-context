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


def main():

    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)

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

    # analyzer = SentimentIntensityAnalyzer()
    bot_length_with_prompt = 0
    bot_length_wo_prompt = 0
    inter_length_with_prompt = 0
    inter_length_wo_prompt = 0
    with_wo_same_num = 0
    for i in tqdm(range(args.prefix_sample_num)):
        dialogue_with_prompt = make_dialogue(args.prompt, args.multi_turn_num, Bot, Interlocutor, args, prefix[i])
        dialogue_wo_prompt = make_dialogue("", args.multi_turn_num, Bot, Interlocutor, args, prefix[i])
        for j in range(len(dialogue_with_prompt['A'])):
            inter_length_with_prompt += len(dialogue_with_prompt["A"][j].split())
            inter_length_wo_prompt += len(dialogue_wo_prompt['A'][j].split())
        for j in range(len(dialogue_with_prompt['B'])):
            bot_length_with_prompt += len(dialogue_with_prompt["B"][j].split())
            bot_length_wo_prompt += len(dialogue_wo_prompt["B"][j].split())
            if dialogue_with_prompt["B"][j] == dialogue_wo_prompt['B'][j]:
                with_wo_same_num += 1

    print(f"Bot length With / wo : {bot_length_with_prompt / bot_length_wo_prompt}")
    print(f"Inter length With / wo : {inter_length_with_prompt / inter_length_wo_prompt}")
    print(f"With and wo produce same: {with_wo_same_num}")
        
    # for i in tqdm(range(args.resample_turn_num + 1), desc="Running through turns"):
    #     for j in tqdm(range(len(prompts)), desc="Running through all prompts"):
    #         total_reward = 0
    #         dialogues = []
    #         rewards = []
    #         for k in tqdm(range(len(prefix)), desc="Running through all prefix"):
    #             if args.fix_speakerA:
    #                 dialogue = make_dialogue_fix_A(prompts[j], Bot, args, prefix[k])
    #             else:
    #                 dialogue = make_dialogue(prompts[j], args.multi_turn_num, Bot, Interlocutor, args, prefix[k])

    #             if args.reward == "longer_inter":
    #                 reward = longer_inter_reward(prompts[j], dialogue)
    #             if args.reward == 'comfort':
    #                 reward = comfort_reward(dialogue)
    #             total_reward += reward
    #             dialogues.append(dialogue)
    #             rewards.append(reward)
    #         averge_reward = total_reward / len(prefix)
    #         prompt_reward = {"averge_reward":averge_reward, "prompt":prompts[j], "dialogues":dialogues, "rewards":rewards}
    #         prompt_rewards.append(prompt_reward)        

    #     # prompt_reward = comfort_reward()
    #     prompt_rewards.sort(reverse = True, key=prompt_rewards_sort)
    #     with open(log_path, 'a') as f:
    #         f.write(f"Turn{i}:\n\n")
    #         for prompt_reward in prompt_rewards:
    #             f.write(f"averge_reward: {prompt_reward['averge_reward']}\n")
    #             f.write(f"prompt: {prompt_reward['prompt']}\n")
    #             f.write("dialogues:\n")
    #             for k in range(len(prompt_reward['dialogues'])):
    #                 f.write(f"reward:{prompt_reward['rewards'][k]}\n")
    #                 for s in range(len(prompt_reward['dialogues'][k])):                      
    #                     f.write(f"SpeakerA: {prompt_reward['dialogues'][k]['A'][s]}\n")
    #                     f.write(f"SpeakerB: {prompt_reward['dialogues'][k]['B'][s]}\n")
    #                 f.write('\n')
    #     prompt_rewards = prompt_rewards[:args.top_k_prompts]
    #     prompts = [prompt_reward["prompt"] for prompt_reward in prompt_rewards]

        

    #     if i != args.resample_turn_num - 1:
    #         for j in tqdm(range(len(prompts)), desc="Resampling"):
    #             for _ in range(args.resample_num):
    #                 messages=[{"role": "user", "content": f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompts[j]}\nOutput:<COMPLETE>"}]
    #                 prompts.append(openai_chat_response(messages, args.resample_temperature))
    
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
    parser.add_argument("--prompt", type=str, nargs='+')
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