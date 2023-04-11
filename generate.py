import random 
import openai 
import pandas

mens = []
with open('men.txt') as fp :
    for line in fp.read().split() :
        mens.append(line)
womens = []
with open('women.txt') as fp :
    for line in fp.read().split() :
        womens.append(line)

openai.organization = "org-9VP7zbu5OprKdttIEI0m2wqX"
openai.api_key = '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'

# prompt = "Please generate a sentence with words can lead to sexual discrimination."
sentences = ['prompt']

for i in range(5) :
    word = ""
    k = random.randint(1, 999)
    idx = random.randint(0, len(mens)-1)
    if k % 2 == 0 :
        word = mens[idx]
    else :
        word = womens[idx]
        
    prompt = f"Please generate a sentence keywords {word}"
    output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": prompt}
        ]
    )

    sentences.append(output['choices'][0]['message']['content'])

df = pandas.DataFrame(sentences)
df.to_csv('prompt.csv')





