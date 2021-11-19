debugN=${1:-20480}
# bash eval_ppl.sh 20480
# config

# Run on all BPE models
model_dirs="game_bert_NO_time_embed_BPE \
            game_bert_NO_time_embed_BPE_t0_curr_mask_step_5_mlm_mlen_512 \
            game_bert_NO_time_embed_BPE_t0_curr_mask_step_30_mlm_mlen_512_1.0002_0.0001 \
            game_bert_NO_time_embed_BPE_t0_curr_mask_step_30_mlm_mlen_512_1.001_0.0001 \
            game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_mlmprob0.5 \
            game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_random_mlm_0.15_0.5
            "
python3.6 eval_ppl.py --eval_h5_data_file cn_char_bert_4096_may_test_set.h5 \
          --train_h5_data_file cn_char_bert_4096_origin_timestamp_gamelog.h5 \
          --model_dirs $model_dirs \
          --debugN $debugN \
          --use_bpe 1

# Run on whitespace models
model_dirs="game_bert_NO_time_embed_whitespace"
python3.6 eval_ppl.py --eval_h5_data_file bert_4096_may_test_set.h5 \
          --train_h5_data_file bert_4096_origin_timestamp_gamelog.h5 \
          --model_dirs $model_dirs \
          --debugN $debugN \
          --use_bpe 0
