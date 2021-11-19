import argparse
import sys
import ipdb
import copy
import h5py
import random

sys.path.append('..')

from game_seq_embedder import EmbedderChecker
from game_seq_embedder import init_behavior_sequence_embedder
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# [BPE, No Use Time Embedding]
# python check_embedder.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_NO_time_embed_whitespace_DEBUG100
# python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/game_bert/bert_model/game_bert_NO_time_embed_whitespace

"""

# Load from hdf5
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_whitespace \
                           --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/game_bert/data/bert_2048_timestamp_gamelog.h5 \
                           --max_n 100

# Load from default txt, NO time embed, white space
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_whitespace \
                         --max_n 200

# Load from default txt, time embed, white space
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_whitespace \
                         --max_n 200

# Load from default txt, NO time embed, BPE
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_BPE \
                         --max_n 200
                         
# Check for sin embedding
python check_embedder.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_sin_whitespace_DEBUG100/ \
                         --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5 \
                         --max_n 200

# Check for seperate design id 
python check_embedder.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_sin_whitespace_sep_des_id_DEBUG100/ \
                         --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5 \
                         --max_n 200

# Check for longformer
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_longformer_NO_time_embed_whitespace/ \
                         --max_n 200

# Check Time gate
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_sin_whitespace_sep_des_id_time_gate/ \
                         --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5 \
                         --max_n 200

# Check design/time gate
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_sin_whitespace_sep_des_id_design_gate_time_gate/ \
                         --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5 \
                         --max_n 200
                         
# Check multi-task
python check_embedder.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_sin_whitespace_sep_des_id_t0_t1_t2_t3_t4_curr_mask_step_40/ \
                         --hdf5_file_path /media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5 \
                         --max_n 200
"""


def game_seq_perturb_func(
        sample: List[str],
        vocab: List,
        perturb_n: int):
    valid_indices = list(range(len(sample)))[::3]
    perturbed_samples = []
    for i in range(perturb_n):
        sample_copy = copy.deepcopy(sample)
        new_word = random.choice(vocab)
        subsitute_index = random.choice(valid_indices)
        # sample_copy[subsitute_index] = new_word
        # print(f"subsitute {new_word} for {sample[subsitute_index]}")
        perturbed_samples.append(sample_copy)
    return perturbed_samples


def game_seq_create_vocab(input: List[List[str]]):
    vocab = set()
    for sample in input:
        vocab.update(set(sample[::3]))  # only use behavior id
    vocab = sorted(list(vocab))
    return vocab


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--hdf5_file_path', type=str)
    parser.add_argument('--max_n', type=int, default=500)

    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    checker = EmbedderChecker()

    # load samples from hdf5 file

    if args.hdf5_file_path:
        hdf5_data = h5py.File(args.hdf5_file_path, 'r')
        new_samples = []
        for dataset_name, data in hdf5_data.items():
            print(f"Load from {dataset_name}")
            for sample in data:
                if len(new_samples) >= args.max_n:
                    break
                sample = sample.astype(str)
                sample = [x for x in sample if x != '[PAD]']
                new_samples.append(' '.join(sample))
        hdf5_data.close()
    else:
        # load samples from txt
        new_samples = []
        with open('sample.txt', 'r') as f:
            for line in f:
                if line.strip():
                    new_samples.append(line.strip())

    new_samples = [x.split() for x in new_samples]
    new_samples = new_samples[:args.max_n]

    # get embed function
    embedder = init_behavior_sequence_embedder(args.model_dir)
    embed_func = embedder.embed

    # conca_output_tasks = None
    conca_output_tasks = ['task0', 'task1', 'task2', 'task3', 'task4']
    # # checker.check_all(embed_func, new_samples)
    # pass_ed, kmeans_avg_ed, random_avg_ed = checker.edit_distance_check(embed_func, new_samples)
    # print(f"pass_ed: {pass_ed}, kmeans_avg_ed: {kmeans_avg_ed}, random_avg_ed: {random_avg_ed}")

    # check batch unique embedding
    pass_batch_consitent = checker.random_batch_consistent_check(embed_func, new_samples, 20, 5,
                                                                 conca_output_tasks=conca_output_tasks)
    print(f"pass_batch_consitent: {pass_batch_consitent}")

    # clustering check
    ari = checker.clustering_check(embed_func, new_samples, game_seq_create_vocab, game_seq_perturb_func, perturb_n=5)
    print(f"Ari: {ari}")

    # # check distances after clustering
    # result = checker.distance_check(embed_func, new_samples, 20)
    # print(f"Result after clustering: {result}")


if __name__ == '__main__':
    main()
