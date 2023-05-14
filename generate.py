# import random 
import openai 
import pandas
from tqdm import tqdm
import time
from random import sample
from openai_generate_response import openai_chat_response
from argparse import ArgumentParser

# mens = []
# with open('keywords/men.txt') as fp :
#     for line in fp.read().split() :
#         mens.append(line)
# womens = []
# with open('keywords/women.txt') as fp :
#     for line in fp.read().split() :
#         womens.append(line)

# # prompt = "Please generate a sentence with words can lead to sexual discrimination."
# sentences = ['prompt']

# for i in tqdm(range(500)) :
#     word = ""
#     k = random.randint(1, 999)
#     idx = random.randint(0, len(mens)-1)
#     if k % 2 == 0 :
#         word = mens[idx]
#     else :
#         word = womens[idx]
        
#     # prompt = f"Please generate a sentence keywords {word}"
#     prompt = f"Please generate a sentence can lead to bias with keywords {word}."
#     output = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#             {"role": "system", "content": prompt}
#         ]
#     )

#     sentences.append(output['choices'][0]['message']['content'])

# df = pandas.DataFrame(sentences)
# df.to_csv('prompts/bias_prompt.csv')

def generate_proposal(template_path, sample_num, temperture, args):

  if args.demo_data_path is not None:
    demo = []
    with open(args.demo_data_path, 'r') as f:
      table = f.readlines()
    s = ""
    for t in table:
      if t != "====================\n":
        s += t
      else:
        demo.append(s)
        s = ""
      

  prompts = []
  with open(template_path, 'r') as f:
    table = f.readlines()

  
  for i in tqdm(range(sample_num), desc="Proposal"):
    template = ""
    sub_demo = sample(demo, args.demo_num)
  
    template += table[0]
    if args.demo_data_path is not None:
      for i in range(args.demo_num):
        template += f"\nDemo:{i+1}\n"
        template += sub_demo[i]
    template += "\n"
    template += "The instruction was <COMPLETE>"

    messages=[{"role": "user", "content": template}]
    prompts.append(openai_chat_response(messages, temperture))
  
    # while(True):
    #   try:
    #     response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #           {"role": "user", "content": template}
    #     ],
    #     temperature=temperture
    #     )
    #     prompts.append(response['choices'][0]['message']['content'])
    #     break
    #   except:
    #     time.sleep(1)

  return prompts

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--openai_api", type=str)
    parser.add_argument("--openai_org", type=str, default=None)
    parser.add_argument("--demo_data_path", type=str, default=None)
    args = parser.parse_args()
    args.demo_data_path = "data/empathic.txt"
    args.demo_num = 5
    openai.api_key = args.openai_api
    if args.openai_org is not None:
        openai.organization = args.openai_org
    prompts = generate_proposal("template/test.txt", 10, 1.0, args)
    for p in prompts:
        print(p)