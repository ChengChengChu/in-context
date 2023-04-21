


def make_dialogue(prompt, multi_turn_num, Bot, Interlocutor, args, prefix):
  dialogue = []
  dialogue.append(prefix)
  # if prefix is None:
  #   # dialogue.append(Interlocutor.make_response(["<|startoftext|>"]))
  #   dialogue.append(Interlocutor.make_response([""]))
  # else:
  #   dialogue.append(prefix)

  # print(f"prefix:{dialogue[0]}")
  
  for i in range(multi_turn_num):
    bot_response = Bot.make_response(dialogue, prompt)
    # print(f"bot:{bot_response}")
    dialogue.append(bot_response)
    interlocutor_response = Interlocutor.make_response(dialogue, "You objective now is to act like a complainer, you should specific comaplin about something that happened during your daily work. The conversation start from you.")
    # print(f"interlocutor:{interlocutor_response}")
    dialogue.append(interlocutor_response)

  return dialogue