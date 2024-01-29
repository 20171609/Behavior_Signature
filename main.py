import os
import gc
import csv
import pickle
import glob
from datetime import datetime 

import matplotlib.pyplot as plt
from GMM_Quantization import make_gmm
import global_
from utils import *
from profiling import b_profiling
import traceback

def main(dataset_path, attack, change_feature, add_src, confidence, separate_attackIP, count_prot, train_window, test_window, n_components,real_time, make_zero, using_minmax):
    # dataset_path = "CTU-Rbot"\

    # Profiling에 사용
    min_data = 5
    # change_feature = True
    # add_src = True
    # separate_attackIP = False

    # GMM에 사용
    # n_components = 20
    # attack = True
    # confidence = 1.28

    # test할 때 사용
    # seperate = True (얜 GMM도)
    add_victim = False

    train_path = [rf"dataset\{dataset_path}\train\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'train'))]
    test_path = [rf"dataset\{dataset_path}\test\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'test'))]

    global_.initialize(train_path[0], change_feature, attack, separate_attackIP, count_prot, train_window, test_window, using_minmax)

    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})_vic({add_victim})"

    if not os.path.isdir(f"./preprocessing"):
        os.mkdir(f"./preprocessing")

    if not os.path.isdir(f"./preprocessing/{dataset_path}"):
        os.mkdir(f"./preprocessing/{dataset_path}")

    if not os.path.isdir(f'./preprocessing/{dataset_path}/profiling'):
        os.mkdir(f'./preprocessing/{dataset_path}/profiling')

    print("Profiling 시작")
    b_profiling(train_path, "train", parameter, min_data, dataset_path, add_victim)
    print("Profiling 끝")

    print("test profiling 시작")
    b_profiling(test_path, "test", parameter, min_data, dataset_path, add_victim)
    print("Test 끝")

    train_raw = []
    train_key = []

    test_raw = []
    test_key = []

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

    # 'test_feature'로 시작하는 모든 파일 찾기
    test_ffiles = glob.glob(os.path.join(folder, 'test_feature*'))
    test_ffiles.sort()
    for file in test_ffiles:
        with open(file, 'rb') as f:
            test_raw += pickle.load(f)
    
    # 'test_key'로 시작하는 모든 파일 찾기
    test_kfiles = glob.glob(os.path.join(folder, 'test_key*'))
    test_kfiles.sort()
    for file in test_kfiles:
        with open(file, 'rb') as f:
            test_key += pickle.load(f)

    if not os.path.isdir(f'./preprocessing/{dataset_path}/GMM'):
        os.mkdir(f'./preprocessing/{dataset_path}/GMM')

    print(len(train_raw))
    # GMM 이름
    dp_GMM = f"n({n_components})_atk({attack})_conf({confidence})_sepIP({separate_attackIP})_cf({change_feature})_mm({using_minmax})_GMM.pkl"

    # GMM 생성 부분
    if not os.path.isfile(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}"):
        print("GMM 생성 해야함")
        make_gmm(train_raw, train_key, n_components, dp_GMM, dataset_path)

    print(f"{n_components}n {attack}attack GMM 불러옴")
    with open(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}", 'rb') as f:
        pattern_gmm = pickle.load(f)
   
    parameter += f'_pro({count_prot})_as({add_src})_conf({confidence})_n({n_components})_zero({make_zero})'
    
    # ip별 퀀타이제이션 셋 만들기
    if os.path.isfile(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl"):
        with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
            train_data = pickle.load(f)
            
        with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
            test_data = pickle.load(f)
    else:
        parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})"

        train_data = pattern_gmm.transform_tokenize(train_raw, confidence=confidence)
        test_data = pattern_gmm.transform_tokenize(test_raw, confidence=confidence)

        if make_zero:
            for idx, data in enumerate(train_raw):
                train_data_tmp = list(train_data[idx])  # Convert to list if it's a string for mutability
                for idx_feature, d in enumerate(data):
                    if d == 0:
                        id_start, id_end = idx_feature * 2, idx_feature * 2 + 2
                        train_data_tmp[id_start:id_end] = '00'  # Replace the slice with '00'

                train_data[idx] = ''.join(train_data_tmp)  # Convert back to string if needed
                    
            for idx, data in enumerate(test_raw):
                test_data_tmp = list(test_data[idx])  # Convert to list if it's a string for mutability
                for idx_feature, d in enumerate(data):
                    if d == 0:
                        id_start, id_end = idx_feature * 2, idx_feature * 2 + 2
                        test_data_tmp[id_start:id_end] = '00'  # Replace the slice with '00'

                test_data[idx] = ''.join(test_data_tmp)  # Convert back to string if needed

        if add_src:
            #데이터 불러오기
            folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'

            train_src = []
            test_src = []
            # 'train_feature'으로 시작하는 모든 파일 찾기
            train_ffiles_src = glob.glob(os.path.join(folder, 'train_srcflag*'))
            train_ffiles_src.sort()
            for file in train_ffiles_src:
                with open(file, 'rb') as f:
                    train_src += pickle.load(f)

            train_data = [f"{train}{src}" for train, src in zip(train_data, train_src)]
        
            test_ffiles_src = glob.glob(os.path.join(folder, 'test_srcflag*'))
            test_ffiles_src.sort()
            for file in test_ffiles_src:
                with open(file, 'rb') as f:
                    test_src += pickle.load(f)

            test_data = [f"{test}{src}" for test, src in zip(test_data, test_src)]

        if count_prot:
            #데이터 불러오기
            folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'

            train_prot = []
            test_prot = []
            
            # 'train_feature'으로 시작하는 모든 파일 찾기
            train_ffiles_prt = glob.glob(os.path.join(folder, 'train_protflag_*'))
            train_ffiles_prt.sort()
            for file in train_ffiles_prt:
                with open(file, 'rb') as f:
                    train_prot += pickle.load(f)

            train_data = [f"{train}{prt}" for train, prt in zip(train_data, train_prot)]
        
            test_ffiles_prt = glob.glob(os.path.join(folder, 'test_protflag_*'))
            test_ffiles_prt.sort()
            for file in test_ffiles_prt:
                with open(file, 'rb') as f:
                    test_prot += pickle.load(f)

            test_data = [f"{test}{prt}" for test, prt in zip(test_data, test_prot)]

        parameter += f'_pro({count_prot})_as({add_src})_conf({confidence})_n({n_components})_zero({make_zero})'

        if not os.path.isdir(f"./debug_data"):
            os.mkdir(f"./debug_data")

        if not os.path.isdir(f"./debug_data/{dataset_path}"):
            os.mkdir(f"./debug_data/{dataset_path}")

        if not os.path.isdir(f'./debug_data/{dataset_path}/{parameter}'):
            os.mkdir(f'./debug_data/{dataset_path}/{parameter}')
        
        with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'wb') as f:
            pickle.dump(train_data,f)
        with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'wb') as f:
            pickle.dump(test_data,f)
    
    print(len(train_data))
    
    train_multi_dict, train_label, attack_quantization_multi_set = make_quantization_dict(train_data, train_key)    
    
    with open(f"./debug_data/{dataset_path}/{parameter}/train_multi_dict_attack{attack}.pkl", 'wb') as f:
        pickle.dump(train_multi_dict,f)
    
    if train_window and real_time:
            train_multi_dict, train_label = make_quantization_dict_window(train_data, train_key, train_window)
        
    test_multi_dict,test_label = make_quantization_test_dict(test_data, test_key)
    
    with open(f"./debug_data/{dataset_path}/{parameter}/test_multi_dict_attack{attack}.pkl", 'wb') as f:
        pickle.dump(test_multi_dict,f)
    
    if not os.path.isdir(f'./result'):
        os.mkdir(f'./result')

    if not os.path.isdir(f'./result/{dataset_path}'):
        os.mkdir(f'./result/{dataset_path}')

    # evaluate
    print("평가 시작")
    file_name = f"as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-n({n_components})-atk({attack})-conf({confidence})_window({train_window}-{test_window})_zero({make_zero})_mm({using_minmax}).csv"
    save_file = f"./result/{dataset_path}/{file_name}.csv"
    
    if real_time:
        evaluate_realtime(train_multi_dict, train_label, test_data, test_key, save_file)
    else:
        evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file)

    #score 측정

if __name__ == "__main__":
    try:
        for data in ['Test_Victim']:
            for attack in [1]: # 0이 정상 1이 공격 2가 혼합
                for change_feature in [False]:
                    for count_prot in [False]:
                        for seperate_attackIP in [True]:
                            for using_minmax in [True]:
                                for add_src in [True]:
                                    for confidence in [1000000]:
                                        for n_components in [80]:
                                            for real_time in [0]:
                                                for train_window in [0]:
                                                    for test_window in [10]:
                                                        for make_zero in [False]:
                                                            main(data, attack, change_feature, add_src, confidence, seperate_attackIP, count_prot, train_window, test_window, n_components,real_time, make_zero, using_minmax)

    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{add_src} add_src-_attackIP-{seperate_attackIP} test에서 에러 발생\n")
            f.write(f"{error_info}\n\n")