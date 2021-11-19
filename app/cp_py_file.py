def main():
    file_path = '/Users/jiashupu/netease_projects/game_bert/app/pjs_bert_model/bert_model.py'
    save_path = '/Users/jiashupu/netease_projects/game_seq_embedder/game_seq_embedder/custom_models/bert_model.py'

    with open(file_path, 'r') as f_r:
        with open(save_path, 'w') as f_w:
            for line in f_r:
                if line.startswith("from transformers"):
                    new_line = line.replace('from transformers', 'from ..transformers')
                else:
                    new_line = line
                f_w.write(new_line)
    print(f"Copy file from {file_path} to {save_path}")


if __name__ == '__main__':
    main()
