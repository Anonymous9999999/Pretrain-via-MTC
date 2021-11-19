import ipdb
import torch
import collections


def main():
    shape = (10000, 10)
    mask_prob = 0.15
    probability_matrix = torch.full(shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    mask_index_counter = collections.Counter(torch.where(masked_indices == True)[1].tolist())
    values = list(mask_index_counter.values())

    avg_value = torch.mean(torch.tensor(values, dtype=torch.float))
    max_value = max(values)
    min_value = min(values)

    print(mask_index_counter)
    print(f"Avg_value: {avg_value}, max_value: {max_value}, min_value: {min_value}, max_ratio: {min_value / max_value}")


if __name__ == '__main__':
    main()
