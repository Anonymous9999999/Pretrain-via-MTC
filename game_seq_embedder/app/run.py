import argparse
import h5py
import sys
import random
import ipdb

sys.path.append('..')
from game_seq_embedder import init_behavior_sequence_embedder

# [White Space, Use Time Embedding]
# python run.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_whitespace_DEBUG100

# [White Space, No Use Time Embedding]
# python run.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_NO_time_embed_whitespace_DEBUG100

# [BPE, No Use Time Embedding]
# python run.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_NO_time_embed_BPE_DEBUG100

# [BPE, Use Time Embedding]
# python run.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_BPE_DEBUG100

# [BPE, Use Time Embedding]
# python run.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_sin_whitespace_sep_des_id_DEBUG100

# [BPE, Use Time Embedding]
# python run.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_whitespace

# [BPE, Use Time Embedding]
# python run.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_sin_whitespace_sep_des_id_t0_t1_t2_t3_t4

# DEBUG, TEST BPE
# python run.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_BPE


import faulthandler

faulthandler.enable()


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--model_dir', type=str)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    new_samples = []
    with open('sample.txt', 'r') as f:
        for line in f:
            if line.strip():
                new_samples.append(line.strip())

    print(f"Number of samples: {len(new_samples)}")

    # no_mask_embedder = init_behavior_sequence_embedder(args.model_dir)

    mask_embedder = init_behavior_sequence_embedder(args.model_dir,
                                                    mask_multiple_time=2,
                                                    mask_prob=0.5,
                                                    output_mask_embedding=False)

    # special sequence
    sepcial_seq = ['400587', '0', '1587320962', '400587', '0', '1587326294', '400587', '0', '1587327498', '400587', '0',
                   '1587355522', '400587', '0', '1587355860', '400587', '0', '1587358646', '400587', '0', '1587360183',
                   '400587', '0', '1587369201', '400587', '0', '1587369227', '400587', '0', '1587373641', '400587', '0',
                   '1587373853', '400587', '0', '1587373856', '400587', '0', '1587374532', '400587', '0', '1587374534',
                   '400587', '0', '1587374712', '400587', '0', '1587374715', '400587', '0', '1587375114', '400587', '0',
                   '1587375117', '400587', '0', '1587375212', '400587', '0', '1587375215', '400587', '0', '1587375834',
                   '400587', '0', '1587375836', '400587', '0', '1587375907', '400587', '0', '1587375910', '400587', '0',
                   '1587376601', '400587', '0', '1587376604']

    # TODO, 这里考虑一下，最好传一个LIST给我
    new_samples = [x.split() for x in new_samples]

    # embedding = embedder.embed(new_samples, conca_output_tasks=['task0', 'task1', 'task2', 'task3', 'task4'])
    # embedding = embedder.embed(new_samples, conca_output_tasks=['task0', 'task1'])
    # no_mask_embedding = no_mask_embedder.embed(new_samples)
    mask_embedding = mask_embedder.embed([sepcial_seq, new_samples[0]])

    ipdb.set_trace()


if __name__ == '__main__':
    main()
