import argparse
import h5py
import os
import sys
import ntpath
import subprocess

sys.path.append('..')
import pandas as pd
import random
import ipdb
import math

sys.path.append('..')
from game_seq_embedder import init_model_and_tokenizer
from app_utils import load_dataset_from_hdf5_by_indices, get_all_indices, get_dataset, tokenizer_post_process
from game_seq_embedder.transformers import Trainer, DataCollatorForLanguageModeling

"""

目前这个版本还是不支持BPE的

[ubuntu], BERT, white space, no time embed
python3.6 eval_ppl.py --h5_data_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_may_test_set.h5 \
          --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_whitespace \
          --debugN 500

[ubuntu], BERT, BPE, NO time embed
python3.6 eval_ppl.py --h5_data_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/cn_char_bert_4096_may_test_set.h5 \
          --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_BPE \
          --debugN 100

[ubuntu], BERT, BPE, NO time embed
python3.6 eval_ppl.py --eval_h5_data_file cn_char_bert_4096_may_test_set.h5 \
          --train_h5_data_file cn_char_bert_4096_origin_timestamp_gamelog.h5 \
          --model_dirs game_bert_NO_time_embed_BPE game_bert_NO_time_embed_BPE_t0_curr_mask_step_5_mlm_mlen_512 \
          --debugN 100 \
          --use_bpe 1
          
"""


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--train_h5_data_file', type=str)
    parser.add_argument('--eval_h5_data_file', type=str)
    parser.add_argument('--model_dirs', type=str, nargs='+')
    parser.add_argument('--debugN', type=int)
    parser.add_argument('--use_bpe', type=int)
    args = parser.parse_args()
    return args


def get_seq_time_max_index(model, use_time_embed):
    # Get Max sequence Length
    max_sequence_length = None
    for param_name, param in model.named_parameters():
        if 'position_embeddings' in param_name:
            max_sequence_length = param.shape[0]
            break

    # Get Max time index
    max_time_index = None
    if use_time_embed:
        # compute max time index
        for param_name, param in model.named_parameters():
            if 'time_gap_embeddings' in param_name:
                max_time_index = param.shape[0] - 1
                break
        print(f"Max time index: {max_time_index}")
    return max_sequence_length, max_time_index


def create_data_for_computing_ppl(file_path, debugN, use_bpe):
    # eval ppl
    all_indices, total_num = get_all_indices(file_path, debugN)

    if use_bpe:
        is_decode_utf8 = True
    else:
        is_decode_utf8 = False

    # evaluate
    # 这里BPE是通过构建两个文件来实现的, BPE的话是带CN CHAR的
    # TODO, to support other types of data
    eval_batch_data = load_dataset_from_hdf5_by_indices(file_path,
                                                        all_indices,
                                                        is_decode_utf8=is_decode_utf8
                                                        )

    return eval_batch_data


def main():
    args = args_parse()

    ppl_df = {'pretrain_task': [], 'train_ppl': [], 'test_ppl': [], 'train_loss': [], 'test_loss': []}

    host_id = subprocess.check_output('hostid').strip().decode('utf-8')

    if host_id == '007f0101':
        # set embedding cache path
        data_base_dir = '/media/iamlxb3/2D97AD940A9AD661/data/game_bert'
        pretrain_base_dir = '/media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/'
    else:
        data_base_dir = '/root/game_bert/data/sample_processed'
        pretrain_base_dir = '/root/game_bert/bert_model/'

    eval_h5_data_file_path = os.path.join(data_base_dir, args.eval_h5_data_file)
    train_h5_data_file_path = os.path.join(data_base_dir, args.train_h5_data_file)

    # Get data for evaluation
    use_bpe = True if args.use_bpe else False
    eval_batch_data = create_data_for_computing_ppl(eval_h5_data_file_path, args.debugN, use_bpe)
    train_batch_data = create_data_for_computing_ppl(train_h5_data_file_path, args.debugN, use_bpe)
    datas = [(eval_batch_data, 'test'), (train_batch_data, 'train')]

    for model_name in args.model_dirs:

        model_dir = os.path.join(pretrain_base_dir, model_name)

        # (1.) init Model
        model, embed_tokenizer, use_time_embed, use_bpe, use_sinusoidal, seperate_design_id = init_model_and_tokenizer(
            model_dir)
        model_name = model._get_name()
        print(f"[Load Model] {model_name} SUCCESS!")

        max_sequence_length, max_time_index = get_seq_time_max_index(model, use_time_embed)

        if model_name == 'LongformerForMaskedLM':
            max_sequence_length -= 2

        # (2.) Init Tokenizer
        # tokenizer
        if len(embed_tokenizer) == 1:
            embed_tokenizer = embed_tokenizer[0]
            tokenizer = tokenizer_post_process(embed_tokenizer, max_sequence_length, 'bpe' if use_bpe else 'whitespace')
            behave_tokenizer, design_tokenizer = None, None
            data_collator_tokenizer = tokenizer
        else:
            behave_tokenizer, design_tokenizer = embed_tokenizer
            tokenizer = None
            behave_tokenizer = tokenizer_post_process(behave_tokenizer, max_sequence_length, 'whitespace')
            design_tokenizer = tokenizer_post_process(design_tokenizer, max_sequence_length, 'whitespace')
            data_collator_tokenizer = behave_tokenizer

        ppl_df['pretrain_task'].append(ntpath.basename(model_dir))

        for data, data_type in datas:

            # 这里是通过tokenizer来区分是否是seperate id的
            eval_dataset = get_dataset(tokenizer,
                                       max_sequence_length,
                                       behave_tokenizer=behave_tokenizer,
                                       design_tokenizer=design_tokenizer,
                                       use_time_embed=use_time_embed,
                                       debugN=args.debugN,
                                       hdf_data=data,
                                       use_bpe=use_bpe,
                                       max_time_gap=max_time_index,
                                       use_sinusoidal=use_sinusoidal)

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=data_collator_tokenizer, mlm=True, mlm_probability=0.15
            )
            data_collator.use_time_embed = use_time_embed
            data_collator.seperate_design_id = seperate_design_id

            trainer = Trainer(
                model=model,
                args=None,
                data_collator=data_collator,
                prediction_loss_only=True,
            )

            eval_output = trainer.evaluate(eval_dataset=eval_dataset, use_time_embed=use_time_embed,
                                           seperate_design_id=seperate_design_id)
            perplexity = math.exp(eval_output["eval_loss"])
            print({"perplexity": perplexity})

            if data_type == 'test':
                ppl_df['test_loss'].append(eval_output["eval_loss"])
                ppl_df['test_ppl'].append(perplexity)
            elif data_type == 'train':
                ppl_df['train_loss'].append(eval_output["eval_loss"])
                ppl_df['train_ppl'].append(perplexity)
            else:
                raise Exception

    ppl_df = pd.DataFrame(ppl_df)
    save_path = f'../results/ppl_{use_bpe}_{args.debugN}.csv'
    ppl_df.to_csv(os.path.abspath(save_path), index=False)
    print(f"Save PPL dataframe to {save_path}, df_shape: {ppl_df.shape}")


if __name__ == '__main__':
    main()
