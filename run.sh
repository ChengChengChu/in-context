# python3 main.py \
# --prompt_path prompts/chat_key_500.csv \
# --save_path dialogGPT_random_key.csv \
# --bot DialogGPT 

# python3 main.py \
# --template_path template/comfort.txt \
# # --interlocutor_model_path results/complain_dg_ms/checkpoint-505/ \
# --interlocutor DialogGPT \
# --sample_num 1 \
# --multi_turn_num 2

python3 main.py \
--template_path template/longer_inter.txt \
--sample_num 10 \
--multi_turn_num 2 \
--reward longer_inter \
--resample_turn_num 3 \
--resample_num 3 \
--top_k_prompts 5 \
--demo_data_path data/empathic.txt \
--demo_num 5 \
--prefix_data_path data/complain_prefix.txt