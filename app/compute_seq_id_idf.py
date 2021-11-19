import os
import ipdb
import json
import math
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from game_seq_embedder import init_model_and_tokenizer
from app_utils import get_all_indices
from app_utils import hdf5_load_dataset


def load_save_json(json_path, mode, verbose=1, encoding='utf-8', data=None):
    if mode == 'save':
        assert data is not None
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose >= 1:
                print(f"save json data to {json_path}")
    elif mode == 'load':
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding=encoding) as f:
                response = json.load(f)
            if verbose >= 1:
                print(f"load json from {json_path} success")
        else:
            raise Exception(f"{json_path} does not exist!")
        return response
    else:
        raise NotImplementedError


def bpe_tokenzie(seq, tokenizer):
    game_ids = []
    for id_i, game_id in enumerate(seq):
        if (id_i + 1) % 3 != 0:
            game_ids.append(game_id)
    bpe_tokenzied_data = ''.join(
        [tokenizer.game_id_cn_char_map.get(x, tokenizer.game_id_cn_char_map['[UNK]']) for x in game_ids])
    tokenizer_output = tokenizer.encode(bpe_tokenzied_data)
    bpe_tokenzied_tokens = tokenizer_output.tokens
    bpe_tokenzied_ids = tokenizer_output.ids
    return bpe_tokenzied_tokens, bpe_tokenzied_ids


def main():
    host_id = subprocess.check_output('hostid').strip().decode('utf-8')
    if host_id == '007f0101':
        # h5_data_file_path = '../data/sample_processed/bert_4096_origin_timestamp_gamelog_debug500.h5'
        h5_data_file_path = '/media/iamlxb3/2D97AD940A9AD661/data/game_bert/bert_4096_origin_timestamp_gamelog.h5'
        model_dir = '/media/iamlxb3/2D97AD940A9AD661/model_ckpts/game_bert/' \
                    'game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_mlmprob-0.16_mask_type-normal_t-0.5_alpha-0.5_decay-0.998_DEBUG200000'
    else:
        h5_data_file_path = '../data/sample_processed/bert_4096_origin_timestamp_gamelog.h5'
        model_dir = '../bert_model/game_bert_NO_time_embed_BPE_t0_mlm_mlen_512_random_mlm_0.15_0.5'

    # debug_N = 20000
    debug_N = 100  # 20000
    h5_large_batch = 80000
    idf_save_path = f'../static/bpe_token_idf_debug-{debug_N}.json'
    tf_idf_save_path = f'../static/bpe_token_tfidf_debug-{debug_N}.json'
    idf_avg_count_save_path = f'../static/bpe_token_idf_avg_count_debug-{debug_N}.json'
    idf_sum_count_save_path = f'../static/bpe_token_idf_sum_count_debug-{debug_N}.json'
    token_count_save_path = f'../static/bpe_token_count_debug-{debug_N}.csv'

    log_id_trans_dict_path = '../static/log_object_id_translation.json'
    log_id_trans_dict = load_save_json(log_id_trans_dict_path, 'load')

    all_indices, total_num = get_all_indices(h5_data_file_path,
                                             debug_N,
                                             is_print_pad_ratio=False,
                                             is_shuffle=True)
    step_size = math.ceil(total_num / h5_large_batch)
    seq_data = []
    _, tokenizer, *_ = init_model_and_tokenizer(model_dir)
    tokenizer = tokenizer[0]

    re_cn_char_map = {v: k for k, v in tokenizer.game_id_cn_char_map.items()}
    index_cn_char_map = {v: k for k, v in tokenizer.get_vocab().items()}

    print(f"Step_size: {step_size}")
    for large_batch_i, large_batch_data in enumerate(hdf5_load_dataset(h5_data_file_path,
                                                                       all_indices,
                                                                       step_size,
                                                                       large_batch=h5_large_batch,
                                                                       is_decode_utf8=False)):
        for seq in tqdm(large_batch_data, total=len(large_batch_data), desc='Collect & tokenize data'):
            seq = seq[seq != '[PAD]']
            bpe_tokenzied_tokens, bpe_tokenzied_ids = bpe_tokenzie(seq, tokenizer)
            bpe_tokenzied_ids = [str(x) for x in bpe_tokenzied_ids]
            seq_data.append(' '.join(bpe_tokenzied_ids))

    vectorizer = TfidfVectorizer()
    _ = vectorizer.fit_transform(seq_data)
    bpe_token_ids = vectorizer.get_feature_names()
    idf_values = vectorizer.idf_
    max_idf_value = np.max(idf_values)
    bpe_token_idf_dict = dict(zip(bpe_token_ids, idf_values))

    tf_idf_dict = collections.defaultdict(lambda: [])
    idf_count_dict = collections.defaultdict(lambda: [])
    token_count = collections.defaultdict(lambda: 0)
    token_count_df = {'token_index': [], 'token_log_id': [], 'token_translated': [], 'idf': [], 'count': []}

    for seq in tqdm(seq_data, total=len(seq_data), desc='Computing tfidf'):
        tf_counter = collections.Counter(seq.split(' '))
        for token_index, tf in tf_counter.items():
            idf = bpe_token_idf_dict.get(token_index, max_idf_value)
            tf_idf = tf * idf
            tf_idf_dict[token_index].append(tf_idf)
            idf_count_dict[idf].append(tf)
            token_count[token_index] += tf

    idf_avg_count_dict = {k: float(np.average(v)) for k, v in idf_count_dict.items()}
    idf_sum_count_dict = {k: int(np.sum(v)) for k, v in idf_count_dict.items()}

    for token_index, token_count in token_count.items():
        token_cn_char = index_cn_char_map.get(int(token_index), '')
        token_raw_log_id = [re_cn_char_map.get(x, re_cn_char_map['嗯']) for x in token_cn_char]
        translated_log_id = [log_id_trans_dict[x] for x in token_raw_log_id if x in log_id_trans_dict]
        token_count_df['token_index'].append(token_index)
        token_count_df['token_log_id'].append(tuple(token_raw_log_id))
        token_count_df['token_translated'].append(tuple(translated_log_id))
        token_count_df['idf'].append(bpe_token_idf_dict.get(token_index, -1))
        token_count_df['count'].append(token_count)
    token_count_df = pd.DataFrame(token_count_df)
    token_count_df = token_count_df.sort_values(by='count', ascending=False)
    token_count_df.to_csv(token_count_save_path, index=False)
    print(f"Save token count to {token_count_save_path}")

    avg_tf_idf_dict = {}
    for label, tf_idfs in tf_idf_dict.items():
        avg_tf_idf_dict[label] = np.average(tf_idfs)
    tf_idf_values = np.array(list(avg_tf_idf_dict.values()))

    print(f"Max idf value: {np.max(idf_values)}, "
          f"min idf value: {np.min(idf_values)},"
          f" avg idf value: {np.average(idf_values)},"
          f" dict size: {len(bpe_token_idf_dict)}")

    print(f"Max avg-tfidf value: {np.max(tf_idf_values)}, "
          f"min avg-tfidf value: {np.min(tf_idf_values)},"
          f" avg avg-tfidf value: {np.average(tf_idf_values)},"
          f" dict size: {len(avg_tf_idf_dict)}")

    load_save_json(idf_save_path, 'save', data=bpe_token_idf_dict)
    load_save_json(tf_idf_save_path, 'save', data=avg_tf_idf_dict)
    load_save_json(idf_avg_count_save_path, 'save', data=idf_avg_count_dict)
    load_save_json(idf_sum_count_save_path, 'save', data=idf_sum_count_dict)

    # from sklearn.feature_extraction.text import TfidfVectorizer
    # corpus = [
    #     'This is the first document.',
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'Is this the first document?']
    #
    #
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(corpus)
    # ipdb.set_trace()
    # print(vectorizer.get_feature_names())


if __name__ == '__main__':
    main()
