import argparse
import h5py
import sys
from torch import optim
import random
import ipdb

sys.path.append('..')
from game_seq_embedder import init_behavior_sequence_embedder


# [BPE, Use Time Embedding]
# python run_finetune.py --model_dir /home/iamlxb3/temp_rsync_dir/game_bert/bert_model/game_bert_time_embed_BPE_DEBUG100
# python run_finetune.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_NO_time_embed_whitespace --max_seq_length 126
# python run_finetune.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_bert_time_embed_whitespace --max_seq_length 126
# python run_finetune.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_longformer_NO_time_embed_whitespace --max_seq_length 126
# python run_finetune.py --model_dir /media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/game_reformer_NO_time_embed_whitespace --max_seq_length 126


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--max_seq_length', type=int)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    # Load Data
    new_samples = []
    with open('sample.txt', 'r') as f:
        for line in f:
            if line.strip():
                new_samples.append(line.strip())
    new_samples = [x.split() for x in new_samples]
    if args.max_seq_length:
        assert args.max_seq_length % 3 == 0.0
        print(f"Truncation all samples to length :{args.max_seq_length}")
        new_samples = [x[:args.max_seq_length] for x in new_samples]

    print(f"Number of samples: {len(new_samples)}")
    max_seq_len = max([len(x) for x in new_samples])

    embedder = init_behavior_sequence_embedder(args.model_dir, is_finetune=True)

    embeddings = embedder.embed(new_samples, batch_size=4)

    # training code simple example ...
    optimizer = optim.SGD(embedder.model_params, lr=0.01, momentum=0.9)
    optimizer.zero_grad()
    loss = sum(embeddings.flatten())
    loss.backward()
    optimizer.step()
    ipdb.set_trace()

    # Don't forget to set to feature extraction mode after training, and set the layer to -1
    embedder.set_to_feature_extration_mode()
    embeddings = embedder.embed(new_samples, layer=-1)


if __name__ == '__main__':
    main()
