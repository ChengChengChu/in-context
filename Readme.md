# APE on Chatbot
This repository aims to implement APE on chatbot. You can find the paper of APE [**here**](https://arxiv.org/abs/2211.01910).

You can execute the code by running:
```
python3 main.py \
--proposal_template_path template/general.txt \
--sample_num 20 \
--reward comfort \
--resample_turn_num 1 \
--resample_num 1 \
--top_k_prompts 10 \
--demo_data_path data/empathic_comma.txt \
--demo_num 10 \
--prefix_data_path data/delta_test.txt \
--openai_api $openai_api \
--openai_org $openai_org \
--fix_speakerA \
--prefix_sample_num 5
```

- sample_num: The number of prompts to make when proposaling
- resample_turn_num: The number of turn to resample
- resample_num: The number of prompts to resample per prompt
- top_k_prompts: Using top k prompts to resample
- demo_num: Number of demos use to generate prompt
- prefix_sample_num: Number of dialogue to calculate reward per prompt
- openai_api: Enter your openai api here
- openai_org: Only need if your api requires you to fill in organization

The results will be locate at results/

You can test the prompt using test.py, by running:

```
python3 test.py \
--prompt "I gave my friend instructions to act as SpeakerB in the given demonstrations and to respond empathetically to SpeakerA's experiences, feelings, and situations. My instructions were to express understanding, provide support, ask questions to understand the situation better, and offer words of encouragement or comfort when suitable." "You should not speak too long sentences" \
--multi_turn_num 3 \
--prefix_data_path data/delta_test.txt \
--openai_api $openai_api \
--openai_org $openai_org \
--prefix_sample_num 10
```

- prompt: The prompt that give to chatgpt, you can add more than more prompt, it will prepend at messages with the form, {"role":"system", "content":[prompt1]}, {"role":"system", "content":[prompt2]}...

