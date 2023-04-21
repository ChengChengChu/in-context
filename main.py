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
from dialogue import make_dialogue
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter(action='ignore', category=FutureWarning)

def prompt_rewards_sort(e):
  return e["reward"]

def main():
    
    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)

    fix_seed(args)
    
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


    # analyzer = SentimentIntensityAnalyzer()
    
    prompts = generate_proposal(args.template_path, args.sample_num, args.proposal_temperature, args)
    # prompts = ["prompt" for _ in range(args.sample_num)]
    prompt_rewards = []

    if args.prefix_data_path is not None:
      prefix = []
      with open(args.prefix_data_path, 'r') as f:
        table = f.readlines()
      s = ""
      for t in table:
        if t != "====================\n":
          s += t
        else:
          prefix.append(s)
          s = ""
    
    for i in tqdm(range(args.resample_turn_num), desc="Turn"):
      for j in tqdm(range(len(prompts)), desc="Generating dialogues"):
        dialogue = make_dialogue(prompts[j], args.multi_turn_num, Bot, Interlocutor, args, prefix[8])
        
        if args.reward == "longer_inter":
          prompt_reward = longer_inter_reward(prompts[j], dialogue)
        
        prompt_rewards.append(prompt_reward)        

        # prompt_reward = comfort_reward()
      prompt_rewards.sort(reverse = True, key=prompt_rewards_sort)
      prompt_rewards = prompt_rewards[:args.top_k_prompts]
      prompts = [prompt_reward["prompt"] for prompt_reward in prompt_rewards]

      if i != args.resample_turn_num - 1:
        for j in tqdm(range(len(prompts)), desc="Resampling"):
          for _ in range(args.resample_num):
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                    {"role": "user", "content": f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompts[j]}\nOutput:<COMPLETE>"}
              ],
              temperature=args.resample_temperature
            )
            prompts.append(response['choices'][0]['message']['content'])
    
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
    parser.add_argument("--template_path", type=str, default="")
    parser.add_argument("--reward", type=str, default="comfort")
    parser.add_argument("--prefix_data_path", type=str, default=None)
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