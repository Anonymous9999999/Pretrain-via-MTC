import argparse
import torch
import h5py
import sys
import random
import ipdb

sys.path.append('..')
from game_seq_embedder import init_behavior_sequence_embedder

# DEBUG, TEST BPE
# python check_mask_prob_valid.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_BPE


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

    no_mask_embedder = init_behavior_sequence_embedder(args.model_dir)

    mask_embedder_no_mask_output = init_behavior_sequence_embedder(args.model_dir,
                                                                   mask_multiple_time=2,
                                                                   mask_prob=0.0,
                                                                   output_mask_embedding=False)

    mask_embedder_mask_output = init_behavior_sequence_embedder(args.model_dir,
                                                                mask_multiple_time=2,
                                                                mask_prob=0.0,
                                                                output_mask_embedding=True)

    # TODO, 这里考虑一下，最好传一个LIST给我
    new_samples = [x.split() for x in new_samples]
    # embedding = embedder.embed(new_samples, conca_output_tasks=['task0', 'task1', 'task2', 'task3', 'task4'])
    # embedding = embedder.embed(new_samples, conca_output_tasks=['task0', 'task1'])
    no_mask_embedding = no_mask_embedder.embed(new_samples)
    mask_embedding_no_mask_output = mask_embedder_no_mask_output.embed(new_samples)
    mask_embedding_mask_output = mask_embedder_mask_output.embed(new_samples)

    assert int(torch.sum(mask_embedding_no_mask_output - no_mask_embedding)) <= 1e-6
    assert int(torch.sum(mask_embedding_mask_output - no_mask_embedding)) <= 1e-6

    print(f"All assertion pass!!!")

    ipdb.set_trace()


if __name__ == '__main__':
    main()
