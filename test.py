import openai
from tqdm import tqdm

openai.organization = "org-9VP7zbu5OprKdttIEI0m2wqX"
openai.api_key = "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"

prompt = "Please generate "
# prompt = "Please generate one example of complaining conversation examples spoken by Speaker A and     Speaker B, with Speaker A being the complaier"

fp = open('test_case.txt', 'w')
num_examples = 1000

for i in tqdm(range(num_examples)) :
        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": prompt},
                # {"role": "user", "content": "Why should DevOps engineer learn kubernetes?"},
            ],
        temperature = 1.0
    )


# print(response)
    result = ''
    for choice in response.choices:
        result += choice.message.content

    fp.write(result)

    fp.write('\n' + '=' * 100 + '\n')
fp.close()
# print(result)
