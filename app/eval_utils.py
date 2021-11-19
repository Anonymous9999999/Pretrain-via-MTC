from game_dataset.bpe_dataset import BpeTextDataset
from game_tokenizer.game_tokenizer import create_func1, create_func2


def get_dataset(
        hdf5_file_path,
        tokenizer,
        block_size,
        use_time_embed=False,
        debugN=None,
        hdf_data=None,
):
    """
    目前仅支持BPE类型的数据集
    """
    return BpeTextDataset(
        hdf_data=hdf_data,
        hdf5_file_path=hdf5_file_path,
        tokenizer=tokenizer,
        block_size=block_size,
        use_time_embed=use_time_embed,
        debugN=debugN,
    )


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
