import importlib

mens, womens = [], []
men_keys_to_idx, women_keys_to_idx = {}, {}

# with open('./keywords/men.txt') as fp :
#     idx = 0
#     for line in fp.read().splitlines() :
#         mens.append(line.lower())
#         men_keys_to_idx[line.lower()] = idx
#         idx += 1

# with open('./keywords/women.txt') as fp : 
#     idx = 0
#     for line in fp.read().splitlines() :
#         womens.append(line.lower())
#         women_keys_to_idx[line.lower()] = idx
#         idx += 1

def replace_sentence(sens):
    ''' This function returns two sentences correspond to the given sentence
    str --> str, str
    e.g. 
    He is my father  --> He is my father, She is my mother
    '''
    ret_1 = " "
    ret_2 = " "

    key_word_idx = []

    sens = sens.replace('\n', '') + '\n'

    sens_without_period = []
    
    sens = [x.lower() for x in sens.split()]

    period = [',', '.', '!', '?', '<', '>', '~', '{', '}', '[', ']', "'", '"', ':']
    for s in sens:
        s_ = s
        for p in period:
            s_ = s_.replace(p, '')
        sens_without_period.append(s_)

    assert(len(sens_without_period) == len(sens))

    # find key word list 
    for i in range(len(sens_without_period)) : 
        # print(sens_without_period[i] + '|')
        if sens_without_period[i] in mens or sens_without_period[i] in womens :
            # print("PASS")
            key_word_idx.append(i)
    
    ret_1 = sens[:]
    ret_2 = sens[:]
    gen = False
    for i in key_word_idx :
        tmp = sens_without_period[i]
        if tmp in womens :
            ret_1[i] = ret_1[i].replace(tmp, mens[women_keys_to_idx[tmp]])
            gen = True
        
        if tmp in mens :
            ret_2[i] = ret_2[i].replace(tmp, womens[men_keys_to_idx[tmp]])
            gen = True
    
    return " ".join(ret_1), " ".join(ret_2), gen

def bias_reward(sentences, bot, analyzer):
        
    score = []
    re_sen = []
    re_res = []

    for j in range(len(sentences)):
        tmp_1, tmp_2, gen = replace_sentence(sentences[j])
            
        if gen == False:
            score.append(0.0) ## more penalty
            re_sen.append([tmp_1, tmp_1])
            re_res.append(['none', 'none'])
        else:
            responses = bot.make_response([tmp_1, tmp_2])
            vs_1 = analyzer.polarity_scores(responses[0])
            vs_2 = analyzer.polarity_scores(responses[1])
            score.append(abs(vs_1['compound'] - vs_2['compound']))
            re_sen.append([tmp_1, tmp_2])
            re_res.append([responses[0][0], responses[1][0]])
        # print("=================")
        # print(score, '\n')
    return score, re_sen, re_res

def comfort_reward(dialogue):
    Reward_func = importlib.import_module(".module", "bots").agent
    reward_func = Reward_func()

    s = ""

    for i in range(len(dialogue)):
        s += f"SpeakerA: {dialogue['A'][i]}\n"
        s += f"SpeakerB: {dialogue['B'][i]}\n"
    for i in range(3):
        reward = reward_func.score(s)
        if reward >= 0:
            return reward
    return 0
    

def longer_inter_reward(prompt, dialogue):
  length = 0
  for i in range(len(dialogue)):
    if i != 0 and i % 2 == 0:
      length += len(dialogue[i].split())

  return {"reward":length, "prompt":prompt, "dialogue":dialogue}


