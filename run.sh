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

# python3 main.py \
# --proposal_template_path template/general.txt \
# --sample_num 100 \
# --reward comfort \
# --resample_turn_num 2 \
# --resample_num 3 \
# --top_k_prompts 25 \
# --demo_data_path data/empathic_comma.txt \
# --demo_num 5 \
# --prefix_data_path data/delta_test.txt \
# --openai_api $openai_api \
# --openai_org $openai_org \
# --fix_speakerA \
# --prefix_sample_num 5

python3 main.py \
--proposal_template_path template/general.txt \
--sample_num 1 \
--reward comfort \
--resample_turn_num 1 \
--resample_num 1 \
--top_k_prompts 1 \
--demo_data_path data/empathic_comma.txt \
--demo_num 10 \
--prefix_data_path data/delta_test.txt \
--openai_api $openai_api \
--openai_org $openai_org \
--fix_speakerA \
--prefix_sample_num 1
