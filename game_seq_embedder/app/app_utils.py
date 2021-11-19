import h5py
import collections
import numpy as np

import os
import ipdb
import math
import time
import datetime
import random
import torch

from torch.utils.data.dataset import Dataset


def hdf5_load_dataset(hdf5_file_path, all_indices, step_size, large_batch=1024, is_decode_utf8=False):
    random.shuffle(all_indices)

    for step_i in range(step_size):
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        next_indices = all_indices[step_i * large_batch:(step_i + 1) * large_batch]
        next_data = collections.defaultdict(lambda: [])
        for x in next_indices:
            *dataset_name, index = x.split('_')
            dataset_name = '_'.join(dataset_name)
            next_data[dataset_name].append(int(index))
        large_batch_data = []

        print(f"Load dataset from hdf5 step {step_i}, size next indices: {len(next_indices)}")
        for dataset_name, dataset_indices in next_data.items():

            if dataset_name == 'nsh_2020-04-04':
                print(f"Skip for {dataset_name}")
                continue
            else:
                print(f"Read from {dataset_name} done, size: {len(dataset_indices)}")

            if is_decode_utf8:
                temp_indices_data = hdf5_file[dataset_name][sorted(dataset_indices)]
                temp_indices_data_str = []
                for i, temp_line in enumerate(temp_indices_data):
                    temp_line = [x.decode('utf-8') for x in temp_line]
                    temp_indices_data_str.append(temp_line)
                temp_indices_data_str = np.stack(temp_indices_data_str)
                large_batch_data.append(temp_indices_data_str)
            else:
                large_batch_data.append(hdf5_file[dataset_name][sorted(dataset_indices)])

        large_batch_data = np.concatenate(large_batch_data).astype(str)

        hdf5_file.close()

        yield large_batch_data


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            hdf_data,
            tokenizer,
            block_size: int,
            use_time_embed: bool = False,
            use_bpe=False,
            debugN=None,
            max_time_gap=None,
            use_sinusoidal=False,
            behave_tokenizer=None,
            design_tokenizer=None,
    ):

        if tokenizer is None:
            assert use_time_embed
            assert behave_tokenizer and design_tokenizer
            assert not use_bpe
        self.tokenizer = tokenizer
        self.behave_tokenizer = behave_tokenizer
        self.design_tokenizer = design_tokenizer

        self.use_time_embed = use_time_embed
        self.examples = []
        self.time_gaps = []
        self.design_ids = []

        for sample_i, sample in enumerate(hdf_data):

            if debugN:
                if sample_i >= debugN:
                    print(f"[DEBUG N] Stop loading data, debug N is set to {debugN}")
                    break
            sample = [x for x in sample if x != '[PAD]']
            text_block = ' '.join(sample)
            # --------------------------------------------------------------------------------------------------
            # CODE BY PJS
            # --------------------------------------------------------------------------------------------------

            if use_time_embed:
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                time_gap_block = text_block.split(' ')[2::3]

                # This is only for data assertion
                temp_test_time_gap = time_gap_block[0]
                temp_date_obj = datetime.datetime.fromtimestamp(int(temp_test_time_gap))
                assert 2019 < temp_date_obj.year < 2022

                # compute the time gap if sinusoidal is not used
                if not use_sinusoidal:
                    # TODO, 这个地方我觉得还是要改一下，统一成最大值1024秒，最小单位是秒，但是最小的单位是1秒*100
                    time_gap_block = list(zip(time_gap_block, time_gap_block[1:] + [0]))
                    time_gap_block = [math.ceil((int(t2) - int(t1)) / 100) for t1, t2 in time_gap_block]
                    time_gap_block[-1] = 0
                    assert min(time_gap_block) >= 0

                if tokenizer is not None:
                    time_gap_block = [y for x in zip(time_gap_block, time_gap_block) for y in x][:block_size]
                else:
                    time_gap_block = [y for x in zip(time_gap_block, time_gap_block) for y in x][:int(block_size * 2)]

                # assert len(pure_text_block) == len(time_gap_block)
                if use_bpe:
                    text_block = ''.join(pure_text_block)
                else:
                    text_block = ' '.join(pure_text_block)
            else:
                pure_text_block = [x for i, x in enumerate(text_block.split(' ')) if (i + 1) % 3 != 0]
                if use_bpe:
                    text_block = ''.join(pure_text_block)
                else:
                    text_block = ' '.join(pure_text_block)

            if tokenizer is not None:
                output = tokenizer.encode(text_block)
                tokenized_ids = output.ids
                tokenized_texts = output.tokens

                design_tokenized_ids = None
            else:

                # get behave token
                behave_output = behave_tokenizer.encode(' '.join(text_block.split()[::2]))
                behave_tokenized_ids = behave_output.ids
                behave_texts = behave_output.tokens

                # get design token
                design_output = design_tokenizer.encode(' '.join(text_block.split()[1::2]))
                design_tokenized_ids = design_output.ids
                design_texts = design_output.tokens

                # combine them all
                assert len(behave_tokenized_ids) == len(design_tokenized_ids) == len(behave_texts) == len(design_texts)
                tokenized_ids = behave_tokenized_ids
                tokenized_texts = None

            # if use_bpe:
            #     # assert len(''.join([y for x in output.tokens for y in x]).replace('_', '').replace('▁', '')) == len(
            #     #     text_block), print(f"len of tokenized_texts no equal to origin, text: {tokenized_texts}")
            #     assert len(''.join([y for x in output.tokens for y in x]).replace('_', '').replace('▁', '')) == len(
            #         text_block), print(f"len of tokenized_texts no equal to origin, text: {tokenized_texts}")

            tokenized_ids = tokenized_ids[:block_size]
            if design_tokenized_ids:
                design_tokenized_ids = design_tokenized_ids[:block_size]

            if tokenized_texts:
                tokenized_texts = tokenized_texts[:block_size]

            example = np.array(tokenized_ids)

            if use_time_embed:
                time_gaps = np.array([int(x) for x in time_gap_block], dtype=int)
                if use_bpe:
                    new_time_gaps = []
                    start_index = 0
                    for word in tokenized_texts:
                        word = word.replace('_', '').replace('▁', '')
                        new_time_gap = time_gaps[start_index:start_index + len(word)]
                        new_time_gaps.append(sum(new_time_gap))
                        start_index += len(word)
                    new_time_gaps = np.array(new_time_gaps)
                    time_gaps = new_time_gaps

                # cut off max time gap
                if not use_sinusoidal:
                    time_gaps = np.array([x if x <= max_time_gap - 1 else max_time_gap - 1 for x in time_gaps])
                else:
                    # 这里做一下转换，一天有86400秒
                    time_gap0 = datetime.datetime.fromtimestamp(time_gaps[0])
                    today_start = datetime.datetime(year=time_gap0.year, month=time_gap0.month, day=time_gap0.day)
                    today_start_timestamp = int(time.mktime(today_start.timetuple()))
                    time_gaps = time_gaps - today_start_timestamp

                # recover the length of time gaps for sperate ids
                if tokenizer is None:
                    time_gaps = time_gaps[::2]

                assert example.shape == time_gaps.shape

            if tokenizer is None:
                assert example.shape == time_gaps.shape == np.array(design_tokenized_ids).shape
            # --------------------------------------------------------------------------------------------------

            if len(example) < block_size:

                # pad example
                if tokenizer:
                    all_pad_example = np.full(block_size, tokenizer.pad_token_id)
                else:
                    all_pad_example = np.full(block_size, behave_tokenizer.pad_token_id)
                all_pad_example[:len(example)] = example
                example = all_pad_example

                # pad design_id
                if design_tokenized_ids:
                    all_pad_design_ids = np.full(block_size, design_tokenizer.pad_token_id)
                    all_pad_design_ids[:len(design_tokenized_ids)] = design_tokenized_ids
                    design_tokenized_ids = all_pad_design_ids

                if use_time_embed:
                    all_pad_time_gap = np.full(block_size, 0)
                    all_pad_time_gap[:len(time_gaps)] = time_gaps
                    time_gaps = all_pad_time_gap

            # add example
            self.examples.append(example)

            # add design id
            if not tokenizer:
                self.design_ids.append(np.array(design_tokenized_ids))

            if use_time_embed:
                self.time_gaps.append(time_gaps)

        # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

        # with open(cached_features_file, "wb") as handle:
        #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # logger.info(
        #     "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
        # )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        # Mode: three in one
        if self.tokenizer is None:
            cat_arr = np.concatenate([self.examples[i], self.design_ids[i], self.time_gaps[i]])
            return torch.tensor(cat_arr, dtype=torch.long)
        else:
            # Mode: General
            if self.use_time_embed:
                cat_arr = np.concatenate([self.examples[i], self.time_gaps[i]])
                return torch.tensor(cat_arr, dtype=torch.long)
            else:
                return torch.tensor(self.examples[i], dtype=torch.long)


def get_dataset(
        tokenizer,
        block_size,
        behave_tokenizer=None,
        design_tokenizer=None,
        use_time_embed=False,
        debugN=None,
        hdf_data=None,
        use_bpe=False,
        max_time_gap=None,
        use_sinusoidal=False
):
    return TextDataset(
        hdf_data=hdf_data,
        tokenizer=tokenizer,
        behave_tokenizer=behave_tokenizer,
        design_tokenizer=design_tokenizer,
        block_size=block_size,
        use_time_embed=use_time_embed,
        debugN=debugN,
        use_bpe=use_bpe,
        max_time_gap=max_time_gap,
        use_sinusoidal=use_sinusoidal
    )


def get_all_indices(h5_data_file_path, debug_N):
    hdf5_file = h5py.File(h5_data_file_path, 'r')
    all_indices = []
    all_keys = sorted(hdf5_file.keys())
    total_num = 0
    for key in all_keys:
        data = hdf5_file[key]
        shape = data.shape
        total_num += shape[0]
        all_indices.extend([f'{key}_{x}' for x in range(shape[0])])

    if debug_N:
        all_indices, total_num = all_indices[:debug_N], debug_N
    return all_indices, total_num


def load_dataset_from_hdf5_by_indices(hdf5_file_path, indices, is_decode_utf8=False):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    next_data = collections.defaultdict(lambda: [])
    for x in indices:
        *dataset_name, index = x.split('_')
        dataset_name = '_'.join(dataset_name)
        next_data[dataset_name].append(int(index))
    large_batch_data = []
    for dataset_name, dataset_indices in next_data.items():
        if is_decode_utf8:
            temp_indices_data = hdf5_file[dataset_name][sorted(dataset_indices)]
            temp_indices_data_str = []
            for i, temp_line in enumerate(temp_indices_data):
                temp_line = [x.decode('utf-8') for x in temp_line]
                temp_indices_data_str.append(temp_line)
            temp_indices_data_str = np.stack(temp_indices_data_str)
            large_batch_data.append(temp_indices_data_str)
        else:
            large_batch_data.append(hdf5_file[dataset_name][sorted(dataset_indices)])
    large_batch_data = np.concatenate(large_batch_data).astype(str)
    hdf5_file.close()
    return large_batch_data


def _convert_token_to_id_with_added_voc(token, added_tokens_encoder):
    if token is None:
        return None

    if token in added_tokens_encoder:
        return added_tokens_encoder[token]


def create_func1(sep_token_id, cls_token_id):
    def get_special_tokens_mask(token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [sep_token_id, cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    return get_special_tokens_mask


def create_func2(added_tokens_encoder, mask_token_id):
    def convert_tokens_to_ids(tokens):
        """ Converts a single token, or a sequence of tokens, (str) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if tokens == '[MASK]':
            return mask_token_id

        if tokens is None:
            return None

        if isinstance(tokens, str):
            return _convert_token_to_id_with_added_voc(tokens, added_tokens_encoder)

        ids = []
        for token in tokens:
            ids.append(_convert_token_to_id_with_added_voc(token, added_tokens_encoder))
        return ids

    return convert_tokens_to_ids


def tokenizer_post_process(tokenizer, block_size, type):
    if type == 'whitespace':
        tokenizer.max_len = block_size
        tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
        tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
    elif type == 'bpe':
        tokenizer.max_len = block_size
        tokenizer.cls_token_id = 0
        tokenizer.pad_token_id = 1
        tokenizer.sep_token_id = 2
        tokenizer.unk_token_id = 3
        tokenizer.mask_token_id = 4
        tokenizer.get_special_tokens_mask = create_func1(tokenizer.pad_token_id, tokenizer.cls_token_id)
        tokenizer.added_tokens_encoder = {}
        tokenizer.convert_tokens_to_ids = create_func2(tokenizer.added_tokens_encoder, tokenizer.mask_token_id)
        tokenizer.mask_token = '[MASK]'
        tokenizer._pad_token = '[PAD]'
    else:
        raise NotImplementedError

    return tokenizer
