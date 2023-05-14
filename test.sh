python3 test.py \
--prompt "I gave my friend instructions to act as SpeakerB in the given demonstrations and to respond empathetically to SpeakerA's experiences, feelings, and situations. My instructions were to express understanding, provide support, ask questions to understand the situation better, and offer words of encouragement or comfort when suitable." \
--multi_turn_num 3 \
--prefix_data_path data/delta_test.txt \
--openai_api $openai_api \
--openai_org $openai_org \
--prefix_sample_num 10