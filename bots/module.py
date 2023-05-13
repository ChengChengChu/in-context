import openai
import re
from openai_generate_response import openai_chat_response

class agent() :
    def __init__(self) :        
        
        self.template = "Please read the following conversation about Speaker A and Speaker B. The goal of this task is to rate words Speaker B spoke.\nConversation :\n [CONV]\n(End of conversation fragment)\nHow comforting is Speaker B's words?(on a scale of 1-5, with 1 being the lowest). You cannot say nothing about this conversation."

    def parse_score(self, sentence) :
        sentence = sentence.replace('1-5', '')
        sentence = sentence.replace('out of 5', '')
        sentence = sentence.replace('/5', '')
      
        number_list = re.findall(r'\d+', sentence)
        if len(number_list) > 0 :
            # print("Case 1 : ")
            # print(sentence)
            return float(number_list[0])
        elif "Speaker A" in sentence or "Speaker B" in sentence :
            # print("Case 2 : ")
            # print(sentence)
            return -2
        return -1

    def score(self, conversation) : 
        """
        conversation : string 
        e.g. SpeakerA : aaa
             SpeakerB : bbb

        return type : float(score)
        """
        
        prompt = self.template.replace('[CONV]', conversation)
        # import pdb
        # pdb.set_trace()
        response = openai_chat_response(messages=[{"role" : "system", 'content' : prompt}])
        # response = openai.ChatCompletion.create(
        #            model = 'gpt-3.5-turbo', 
        #            messages=[
        #                 {"role" : "system", 'content' : prompt}
        #                ]
        #         )['choices'][0]['message']['content']

        return self.parse_score(response)


        


