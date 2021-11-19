import h5py
import ipdb


# python3.6 compress_docker_data.py

def main():
    h5_data_file_path = '/media/iamlxb3/2D97AD940A9AD661/data/game_bert/cn_char_bert_4096_origin_timestamp_gamelog.h5'
    save_path = '/home/iamlxb3/temp_rsync_dir/game_bert_docker/data/pre_train/game_player_behavior_sequence.h5'

    hdf5_file = h5py.File(h5_data_file_path, 'r')
    hdf5_file_save = h5py.File(save_path, 'w')

    all_keys = sorted(hdf5_file.keys())
    keep_N_for_each_day = 300

    for key in all_keys:
        data = hdf5_file[key]
        data_kept = data[list(range(keep_N_for_each_day))]
        hdf5_file_save.create_dataset(key,
                                      data=data_kept,
                                      compression="gzip",
                                      fletcher32=True,
                                      chunks=True)
    print(f"Save to {save_path}")


if __name__ == '__main__':
    main()
