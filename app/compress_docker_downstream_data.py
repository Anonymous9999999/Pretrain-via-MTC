import h5py
import ipdb
import hashlib
import random
import numpy as np


# python3.6 compress_docker_downstream_data.py

def main():
    # ------------------------------------------------------------------------------------------------------------------
    # CHURN PREDICTION
    # ------------------------------------------------------------------------------------------------------------------

    h5_data_file_path = '/home/iamlxb3/temp_rsync_dir/game_bert/app/downstream_tasks/data/churn_predict_part_8_2w_debug.h5'
    save_path = '/home/iamlxb3/temp_rsync_dir/game_bert_docker/data/downstream_tasks/churn_prediction.h5'

    hdf5_file = h5py.File(h5_data_file_path, 'r')
    hdf5_file_save = h5py.File(save_path, 'w')

    random.seed(1)

    all_keys = sorted(hdf5_file.keys())
    all_indices = list(range(hdf5_file['date'].shape[0]))
    keep_N = 100
    random_indices = sorted(random.sample(all_indices, keep_N))

    # ['date', 'hand_feature', 'input', 'label']

    print(all_keys)
    for key in all_keys:
        data = hdf5_file[key]
        if key in {'label', 'date', 'hand_feature'}:
            data_kept = data[random_indices]
            # data_kept = data[list(range(keep_N_for_each_day))]
            hdf5_file_save.create_dataset(key,
                                          data=data_kept,
                                          compression="gzip",
                                          fletcher32=True,
                                          chunks=True)
        elif key in {'input'}:
            data_kept = data[random_indices]
            data_flatten = data_kept.flatten()
            new_sequence = []
            for i, x in enumerate(data_flatten):
                if x == b'[PAD]':
                    new_sequence.append(x)
                else:
                    if (i + 1) % 3 != 0:
                        new_sequence.append(hashlib.md5(x).hexdigest().encode('utf-8'))
                    else:
                        new_sequence.append(x)

            input_new_arr = np.array(new_sequence)
            input_new_arr.dtype = 'S32'
            input_new_arr = input_new_arr.reshape((keep_N, 14, 6144))
            # data_kept = data[list(range(keep_N_for_each_day))]
            hdf5_file_save.create_dataset(key,
                                          data=input_new_arr,
                                          compression="gzip",
                                          fletcher32=True,
                                          chunks=True)
        else:
            pass

    print(f"Save to {save_path}")
    # ------------------------------------------------------------------------------------------------------------------

    # # ------------------------------------------------------------------------------------------------------------------
    # # CLUSTERING
    # # ------------------------------------------------------------------------------------------------------------------
    #
    # h5_data_file_path = '/media/iamlxb3/2D97AD940A9AD661/data/game_bert/new_role_id_clustering.h5'
    # save_path = '/home/iamlxb3/temp_rsync_dir/game_bert_docker/data/downstream_tasks/role_id_clustering.h5'
    #
    # hdf5_file = h5py.File(h5_data_file_path, 'r')
    # hdf5_file_save = h5py.File(save_path, 'w')
    #
    # random.seed(1)
    #
    # all_keys = sorted(hdf5_file.keys())
    # all_indices = list(range(hdf5_file['date'].shape[0]))
    # keep_N = 1000
    # random_indices = sorted(random.sample(all_indices, keep_N))
    #
    # # ['date', 'gender_labels', 'grade_labels', 'label', 'mac_labels', 'role_class_labels', 'role_id', 'seq_input']
    #
    # # ['date', 'input', 'label', 'sort_indices']
    # print(all_keys)
    # for key in all_keys:
    #     data = hdf5_file[key]
    #     if key in {'label'}:
    #         data_kept = data[random_indices]
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=data_kept,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    #     elif key in {'seq_input'}:
    #         data_kept = data[random_indices]
    #         data_kept_new = []
    #
    #         for sequence in data_kept:
    #             sequence = sequence.astype(str)
    #             new_sequence = []
    #             for i, x in enumerate(sequence):
    #                 if x == '[PAD]':
    #                     new_sequence.append(x)
    #                 else:
    #                     if (i + 1) % 3 != 0:
    #                         new_sequence.append(hashlib.md5(x.encode('utf-8')).hexdigest())
    #                     else:
    #                         new_sequence.append(x)
    #             data_kept_new.append(new_sequence)
    #
    #         input_new_arr = np.full((len(data_kept_new), 6144), '[PAD]', f'S{32}')
    #         for oneday_i, oneday_ids in enumerate(data_kept_new):
    #             input_new_arr[oneday_i][:len(oneday_ids)] = oneday_ids
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=input_new_arr,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    #     else:
    #         pass
    #
    # print(f"Save to {save_path}")
    # # ------------------------------------------------------------------------------------------------------------------

    # # ------------------------------------------------------------------------------------------------------------------
    # # BUY TIME PREDICTION
    # # ------------------------------------------------------------------------------------------------------------------
    # h5_data_file_path = '/media/iamlxb3/2D97AD940A9AD661/data/game_bert/predict_pay_time_debug5000.h5'
    # save_path = '/home/iamlxb3/temp_rsync_dir/game_bert_docker/data/downstream_tasks/predict_pay_time.h5'
    #
    # hdf5_file = h5py.File(h5_data_file_path, 'r')
    # hdf5_file_save = h5py.File(save_path, 'w')
    #
    # random.seed(1)
    #
    # all_keys = sorted(hdf5_file.keys())
    # print(all_keys)
    # all_indices = list(range(hdf5_file['role_id'].shape[0]))
    # keep_N = 1000
    # random_indices = sorted(random.sample(all_indices, keep_N))
    #
    # # ['item_id', 'item_label', 'label', 'role_id', 'seq_input', 'timegaps']
    #
    # for key in all_keys:
    #     data = hdf5_file[key]
    #     if key in {'timegaps', 'label'}:
    #         data_kept = data[random_indices]
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=data_kept,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    #     elif key in {'role_id'}:
    #         data_kept = data[random_indices]
    #         new_data_kept = []
    #
    #         for i, x in enumerate(data_kept):
    #             new_data_kept.append(hashlib.md5(x).hexdigest().encode('utf-8'))
    #
    #         new_data_kept = np.array(new_data_kept)
    #         new_data_kept.dtype = f'S{32}'
    #
    #         hdf5_file_save.create_dataset(key,
    #                                       data=new_data_kept,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    #     elif key in {'item_id', 'item_label'}:
    #         pass
    #     else:
    #         data_kept = data[random_indices]
    #         data_kept_new = []
    #
    #         for sequence in data_kept:
    #             sequence = sequence.astype(str)
    #             new_sequence = []
    #             for i, x in enumerate(sequence):
    #                 if x == '[PAD]':
    #                     new_sequence.append(x)
    #                 else:
    #                     if (i + 1) % 3 != 0:
    #                         new_sequence.append(hashlib.md5(x.encode('utf-8')).hexdigest())
    #                     else:
    #                         new_sequence.append(x)
    #             data_kept_new.append(new_sequence)
    #
    #         input_new_arr = np.full((len(data_kept_new), 6144), '[PAD]', f'S{32}')
    #         for oneday_i, oneday_ids in enumerate(data_kept_new):
    #             input_new_arr[oneday_i][:len(oneday_ids)] = oneday_ids
    #
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=input_new_arr,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    # print(f"Save to {save_path}")
    # # ------------------------------------------------------------------------------------------------------------------

    # # ------------------------------------------------------------------------------------------------------------------
    # # BOT DETECTION
    # # ------------------------------------------------------------------------------------------------------------------
    # h5_data_file_path = '/home/iamlxb3/temp_rsync_dir/game_bert/app/downstream_tasks/data/bot_detect.h5'
    # save_path = '/home/iamlxb3/temp_rsync_dir/game_bert_docker/data/downstream_tasks/bot_detect.h5'
    #
    # hdf5_file = h5py.File(h5_data_file_path, 'r')
    # hdf5_file_save = h5py.File(save_path, 'w')
    #
    # random.seed(1)
    #
    # all_keys = sorted(hdf5_file.keys())
    # all_indices = list(range(hdf5_file['date'].shape[0]))
    # keep_N = 1000
    # random_indices = sorted(random.sample(all_indices, keep_N))
    #
    # # ['date', 'input', 'label', 'sort_indices']
    #
    # for key in all_keys:
    #     data = hdf5_file[key]
    #     if key in {'date', 'label', 'sort_indices'}:
    #         data_kept = data[random_indices]
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=data_kept,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    #     else:
    #         data_kept = data[random_indices]
    #         data_kept_new = []
    #
    #         for sequence in data_kept:
    #             sequence = sequence.astype(str)
    #             new_sequence = []
    #             for i, x in enumerate(sequence):
    #                 if x == '[PAD]':
    #                     new_sequence.append(x)
    #                 else:
    #                     if (i + 1) % 3 != 0:
    #                         new_sequence.append(hashlib.md5(x.encode('utf-8')).hexdigest())
    #                     else:
    #                         new_sequence.append(x)
    #             data_kept_new.append(new_sequence)
    #
    #         input_new_arr = np.full((len(data_kept_new), 6144), '[PAD]', f'S{32}')
    #         for oneday_i, oneday_ids in enumerate(data_kept_new):
    #             input_new_arr[oneday_i][:len(oneday_ids)] = oneday_ids
    #
    #         # data_kept = data[list(range(keep_N_for_each_day))]
    #         hdf5_file_save.create_dataset(key,
    #                                       data=input_new_arr,
    #                                       compression="gzip",
    #                                       fletcher32=True,
    #                                       chunks=True)
    # print(f"Save to {save_path}")
    # # ------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
