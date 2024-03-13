import os
import gc
import csv
import pickle
import glob
from datetime import datetime 

import matplotlib.pyplot as plt
from GMM_Quantization import make_gmm
from bayesian_block import make_Bayesian
from log_quan import make_log_quan
import global_
from utils import *
from profiling import b_profiling
import traceback
from test import test_live, test_no_live

def main(dataset_path, min_data, attack, change_feature, add_src, separate_attackIP, count_prot, train_window, test_window, n_components, real_time, using_minmax, using_quan, p0, ignore_background, count, live):
    train_path = [rf"dataset\{dataset_path}\train\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'train'))]
    test_path = [rf"dataset\{dataset_path}\test\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'test'))]

    global_.initialize(train_path[0], change_feature, attack, separate_attackIP, count_prot, train_window, test_window, using_minmax)

    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})_ib{ignore_background}_c{count}"

    if not os.path.isdir(f"./preprocessing"):
        os.mkdir(f"./preprocessing")

    if not os.path.isdir(f"./preprocessing/{dataset_path}"):
        os.mkdir(f"./preprocessing/{dataset_path}")

    if not os.path.isdir(f'./preprocessing/{dataset_path}/profiling'):
        os.mkdir(f'./preprocessing/{dataset_path}/profiling')

    print("Profiling 시작")
    b_profiling(train_path, "train", parameter, min_data, dataset_path, ignore_background)
    print("Profiling 끝")

    train_raw = []
    train_key = []

    #데이터 불러오기
    folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'
    
    # 'train_feature'으로 시작하는 모든 파일 찾기
    train_ffiles = glob.glob(os.path.join(folder, 'train_feature*'))

    train_ffiles.sort()
    for file in train_ffiles:
        with open(file, 'rb') as f:
            train_raw += pickle.load(f)
    
    # 'train_key'로 시작하는 모든 파일 찾기
    train_kfiles = glob.glob(os.path.join(folder, 'train_key*'))
    train_kfiles.sort()
    for file in train_kfiles:
        with open(file, 'rb') as f:
            train_key += pickle.load(f)


    if using_quan =='log' and not os.path.isdir(f'./preprocessing/{dataset_path}/LOG'):
        os.mkdir(f'./preprocessing/{dataset_path}/LOG')

    # log 이름
    dp_log = f"log_n({n_components})_atk({attack})_mm({using_minmax})_ib{ignore_background}_cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_{count}c_log.pkl"

    # GMM 생성 부분
    if using_quan =='log':
        if not os.path.isfile(f"./preprocessing/{dataset_path}/LOG/{dp_log}"):
            print("LOG boundary 생성 해야함")
            make_log_quan(train_raw, train_key, dataset_path, n_components,dp_log)

        print(f"log n:{n_components} {attack}attack LOG 불러옴")
        with open(f"./preprocessing/{dataset_path}/LOG/{dp_log}", 'rb') as f:
            pattern_model = pickle.load(f)


    if using_quan == 'log':
        parameter += f'_pro({count_prot})_as({add_src})_log({n_components})'
    

    # train_raw = np.array(train_raw)
    # test_raw = np.array(test_raw)

    # train_raw[:, 7] = 0
    # train_raw[:,12] = 0
    # train_raw[:, 17] = 0
    # train_raw[:, 22] = 0
    
    # test_raw[:, 7] = 0
    # test_raw[:,12] = 0
    # test_raw[:, 17] = 0
    # test_raw[:, 22] = 0

    #     if os.path.isfile(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl"):
    #         with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
    #             train_data = pickle.load(f)
                
    #         with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
    #             test_data = pickle.load(f)
                
    #else:
    train_data = pattern_model.multi_transform(train_raw)

    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})_ib{ignore_background}_c{count}"
    
    
    if add_src:
        #데이터 불러오기
        folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'

        train_src = []
        # 'train_feature'으로 시작하는 모든 파일 찾기
        train_ffiles_src = glob.glob(os.path.join(folder, 'train_srcflag*'))
        train_ffiles_src.sort()
        for file in train_ffiles_src:
            with open(file, 'rb') as f:
                train_src += pickle.load(f)
        print("src : ", len(train_src))
        train_data = [f"{train}{src}" for train, src in zip(train_data, train_src)]
    
        

    if count_prot:
        #데이터 불러오기
        folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'
        train_prot = []
        
        # 'train_feature'으로 시작하는 모든 파일 찾기
        train_ffiles_prt = glob.glob(os.path.join(folder, 'train_protflag_*'))
        train_ffiles_prt.sort()
        for file in train_ffiles_prt:
            with open(file, 'rb') as f:
                train_prot += pickle.load(f)
        print("prot : ", len(train_prot))
        train_data = [f"{train}{prt}" for train, prt in zip(train_data, train_prot)]


    if using_quan == 'log':
        parameter += f'_pro({count_prot})_as({add_src})_log({n_components})'

    if not os.path.isdir(f"./debug_data"):
        os.mkdir(f"./debug_data")

    if not os.path.isdir(f"./debug_data/{dataset_path}"):
        os.mkdir(f"./debug_data/{dataset_path}")

    if not os.path.isdir(f'./debug_data/{dataset_path}/{parameter}'):
        os.mkdir(f'./debug_data/{dataset_path}/{parameter}')
    
    with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'wb') as f:
        pickle.dump(train_data,f)

    file_name = f"log({logN})-as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax})_ib{ignore_background}_c{count}.csv"
    save_file = f"./result/{dataset_path}/{using_quan}_{file_name}.csv"
    
    print(len(train_data))

    if not os.path.isdir(f'./result'):
        os.mkdir(f'./result')

    if not os.path.isdir(f'./result/{dataset_path}'):
        os.mkdir(f'./result/{dataset_path}')

    # evaluate
    print("평가 시작")

    if live:
        train_multi_dict, train_label = make_quantization_dict_live_test(train_data, train_key)    
        test_live(save_file, test_path, min_data, ignore_background, pattern_model, add_src, train_multi_dict, train_label)
    
    else:
        if train_window and real_time:
            train_multi_dict, train_label = make_quantization_dict_window(train_data, train_key, train_window)
        else:
            train_multi_dict, train_label, attack_quan_set = make_quantization_dict(train_data, train_key)
    
        test_no_live(save_file, test_path, parameter, min_data, dataset_path, ignore_background, pattern_model, add_src, count_prot, attack, train_multi_dict,  train_label, attack_quantization_multi_set)


if __name__ == "__main__":
    min_data = 10
    change_feature = False
    seperate_attackIP = False
    count_prot = False
    using_minmax = True
    add_src = True
    real_time = 0
    attack = 1 # 0이 정상 1이 공격 2가 혼합
    train_window = 0
    test_window = 10
    p0 = 0.9
    logN = "test testcode"
    using_quan = 'log'
    count = 111 # 8은 10000이하 제거
    live = True

    try:
        for data in ['CTU-CIC-micro-final']:
            for ignore_background  in [True]:
                main(data, min_data, attack, change_feature, add_src, seperate_attackIP, count_prot, train_window, test_window, logN, real_time, using_minmax, using_quan, p0, ignore_background, count, live)

    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{add_src} add_src-_attackIP-{seperate_attackIP} test에서 에러 발생\n")
            f.write(f"{error_info}\n\n")
