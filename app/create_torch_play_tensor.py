import torch
import ipdb


def main():
    STD = 0.1
    seq_len = 16
    seq_num = 4

    seqs = []
    for i in range(seq_num):
        mean = i
        seq1 = torch.full((seq_len, 768), torch.normal(mean=torch.tensor(mean).float(), std=STD))
        seqs.append(seq1)
    seqs = torch.stack(seqs)
    seqs = torch.transpose(seqs, 1, 0)
    ipdb.set_trace()


if __name__ == '__main__':
    main()
