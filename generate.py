# import random 
import openai 
import pandas
from tqdm import tqdm
import time

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

def generate_proposal(template_path, sample_num, temperture):
    prompts = []
    template = ""
    with open(template_path, 'r') as f:
        table = f.readlines()
        for s in table:
            template += s
    
    for i in tqdm(range(sample_num)):
        while(True):
            try:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                            {"role": "user", "content": template}
                         ],
                temperature=temperture
                )
                prompts.append(response['choices'][0]['message']['content'])
                break
            except:
                time.sleep(1)

    return prompts



if __name__ == "__main__":
    openai.organization = "org-9VP7zbu5OprKdttIEI0m2wqX"
    openai.api_key = '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'
    generate_proposal("template/comfort.txt", 1, 1.0)