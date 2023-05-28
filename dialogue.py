
def make_dialogue(prompt, multi_turn_num, Bot, Interlocutor, args, prefix):
    messages = []
    dialogue = {'A':[], 'B':[]}
    messages.append(prefix['A'][0])
    dialogue['A'].append(prefix['A'][0])
    for i in range(multi_turn_num):
        bot_response = Bot.make_response(messages, prompt)
        messages.append(bot_response)
        dialogue['B'].append(bot_response)
        if i != multi_turn_num - 1:
            inter_response = Interlocutor.make_response(messages)
            dialogue['A'].append(inter_response)
    return dialogue

def make_dialogue_fix_A(prompt, Bot, args, prefix):
    messages = []
    dialogue = {'A':[], 'B':[]}
    for i in range(len(prefix['A'])):
        messages.append(prefix['A'][i])
        bot_response = Bot.make_response(messages, prompt)
        messages.append(bot_response)
        dialogue['A'].append(prefix['A'][i])
        dialogue['B'].append(bot_response)
    return dialogue

