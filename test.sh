python3 test.py \
--prompt "Act as SpeakerB in each of the following demonstrations and provide empathetic and understanding responses to SpeakerA's statements or situations." \
--multi_turn_num 3 \
--prefix_data_path data/delta_test.txt \
--openai_api $openai_api \
--openai_org $openai_org \
--prefix_sample_num 10