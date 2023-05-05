import openai
import time

def openai_chat_response(messages, temperature = 1.0):
    for i in range(10):
      try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature
        )
        return response['choices'][0]['message']['content']
      except:
        time.sleep(i*100)