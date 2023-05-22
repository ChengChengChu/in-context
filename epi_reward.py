import openai
import time, os
from openai_generate_response import openai_chat_response


openai.api_key = ""



def ER_template(s: str) -> str:
    
    tmp = "Emotional reactions is  that expressing emotions such as warmth, compassion, and concern, experienced by peer supporter after reading seeker’s post.\n\n\
Please evaluate the degree of emotional reactions in the following sentence using one of {no emotional reactions, weak emotional reactions, strong emotional reactions} without explanation.\n\n\
Sample:\n\n"

    prompt = tmp + s
    return prompt


def EX_template(s: str) -> str:

    tmp = "Explorations is that improving understanding of the seeker by exploring the feelings and experiences not stated in the post.\n\n\
Please evaluate the degree of explorations in the following sentence using one of {no explorations, weak explorations, strong explorations} without explanation.\n\n\
Sample:\n\n"

    prompt = tmp + s
    return prompt

def IP_template(s: str) -> str:

    tmp = "Define: Interpretations is that communicating an understanding of feelings and experiences inferred from the seeker’s post.\n\n\
Please evaluate the degree of interpretations in the following sentence using one of {no interpretations, weak interpretations, strong interpretations} with no explanation.\n\n\
Sample:\n\n"
    
    prompt = tmp + s
    return prompt

def generate_resposne(prompt, model_name="gpt-4"):
 
    messages=[
                    {"role": "system", "content": "You are a psychologist. Please execute the grading task diligently."},
                    {"role": "user", "content": prompt}
                ]
    return openai_chat_response(messages, temperature=0.7)

def score(s: str) -> int:

    if "no" in s:
        return -1
    elif "weak" in s:
        return 0
    elif "strong" in s:
        return 1
    else: return -10

def get_reward(s: str):

    ip_prompt = IP_template(s)
    ip_reply = generate_resposne(ip_prompt).lower()
    
    ex_prompt = EX_template(s)
    ex_reply = generate_resposne(ex_prompt).lower()

    er_prompt = ER_template(s)
    er_reply = generate_resposne(er_prompt).lower()

    ips = score(ip_reply)
    exs = score(ex_reply)
    ers = score(er_reply)

    return {
        "IP_score": ips,
        "EX_score": exs,
        "ER_score": ers
    }