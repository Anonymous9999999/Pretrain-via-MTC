from game_seq_embedder import init_behavior_sequence_embedder


def main():
    model_dir = 'game_bert_time_embed_whitespace'

    batch_sample = [['400347', '0', '1604578001', '400616', '0', '1604578002'],
                    ['400347', '0', '1604578003', '400347', '0', '1604578004'],
                    ['400000', '0', '1604578037', '400121', '0', '1604578039']]

    embedder = init_behavior_sequence_embedder(model_dir)

    embedding = embedder.embed(batch_sample,
                               batch_size=4,
                               layer=-2)  # output shape: batch_size x 768
    print(embedding.shape)

if __name__ == '__main__':
    main()