import torch
from torch import nn
import openai
from openai_generate_response import openai_chat_response

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """


    def make_response(self, history, prompt = ""):
      messages = []
      if prompt != "":
        messages.append({"role": "system", "content": prompt})
      for i in range(len(history)):
        if i % 2 == 0:
          messages.append({"role": "user", "content": history[i]})
        else:
          messages.append({"role": "assistant", "content": history[i]})
      
      # response = openai.ChatCompletion.create(
      #   model="gpt-3.5-turbo",
      #   messages = messages,
      #   temperature = 0.0
      # )

      # return response['choices'][0]['message']['content']
      return openai_chat_response(messages, 0.0)
    # def make_response(self, prefix_sentences):
        
    #     #openai.api_key = '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'
    #     #openai.api_key =  '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'
        
    #     #openai.api_key =  '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'
    #     with torch.no_grad():
    #         sentences = []
    #         # output_sentences = [tokenizer.encode(x, add_prefix_space=True) for x in output_sentences_string]
    #         # prompt = [tokenizer.encode(x, add_prefix_space=True) for x in first_input_string]
    #         for i in range(len(prefix_sentences)):
                
    #             #total_string  = "There is office in the following response:" + output_sentences_string[i]
    #            # total_string  = "Make the following response full of office:" + output_sentences_string[i]
    #             total_string = prefix_sentences[i]
    #             sentences.append(f"Context: {total_string}\nResponse:")
    #         reply_string = []

    #         # start_sequence = "\nPerson 1:"
    #         # restart_sequence = "\nPerson 2: "

    #         response = openai.Completion.create(
    #             engine="text-ada-001",
    #             prompt=sentences,
    #             temperature=0,
    #             max_tokens=40,
    #             top_p=1,
    #             frequency_penalty=0,
    #             presence_penalty=0.6,
    #             stop=["\n"]
    #             )
    #         for i in range(len(sentences)):

    #             reply_string.append(response['choices'][i]['text'])
    #     # print(reply_string)
    #     # print("response=",reply_string)
    #         # reply_string = tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    #         for i in range(len(reply_string)):
    #             reply_string[i] = [reply_string[i].strip()]

    #     return reply_string
