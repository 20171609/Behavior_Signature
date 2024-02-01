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

def main(dataset_path, min_data, attack, change_feature, add_src, separate_attackIP, count_prot, train_window, test_window, n_components, real_time, using_minmax, using_quan, add_victim, p0):
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

    if using_quan =='gmm' and not os.path.isdir(f'./preprocessing/{dataset_path}/GMM'):
        os.mkdir(f'./preprocessing/{dataset_path}/GMM')
    if using_quan =='log' and not os.path.isdir(f'./preprocessing/{dataset_path}/LOG'):
        os.mkdir(f'./preprocessing/{dataset_path}/LOG')
    if using_quan =='bayesian' and not os.path.isdir(f'./preprocessing/{dataset_path}/Bayesian'):
        os.mkdir(f'./preprocessing/{dataset_path}/Bayesian')

    # GMM 이름
    # dp_GMM = f"n({n_components})_atk({attack})_conf({confidence})_sepIP({separate_attackIP})_cf({change_feature})_mm({using_minmax})_GMM.pkl"
    dp_log = f"log_n({n_components})_atk({attack})_mm({using_minmax})_vic({add_victim})_log.pkl"
    dp_bayes = f"p0({p0})_atk({attack})_mm({using_minmax})_vic({add_victim})_bayes.pkl"

    # GMM 생성 부분
    # if using_quan=='gmm':
    #     if not os.path.isfile(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}"):
    #         print("GMM 생성 해야함")
    #         make_gmm(train_raw, train_key, n_components, dp_GMM, dataset_path)

    #     print(f"{n_components}n {attack}attack GMM 불러옴")
    #     with open(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}", 'rb') as f:
    #         pattern_gmm = pickle.load(f)
    if using_quan =='log':
        if not os.path.isfile(f"./preprocessing/{dataset_path}/LOG/{dp_log}"):
            print("LOG boundary 생성 해야함")
            make_log_quan(train_raw, train_key, dataset_path, n_components,dp_log)

        print(f"log n:{n_components} {attack}attack LOG 불러옴")
        with open(f"./preprocessing/{dataset_path}/LOG/{dp_log}", 'rb') as f:
            pattern_model = pickle.load(f)

    elif using_quan == 'bayesian':
        if not os.path.isfile(f"./preprocessing/{dataset_path}/Bayesian/{dp_bayes}"):
            print("LOG boundary 생성 해야함")
            make_Bayesian(train_raw, train_key, p0, dp_bayes, dataset_path)

        print(f"p0:{p0} {attack}attack Bayesian 불러옴")
        with open(f"./preprocessing/{dataset_path}/Bayesian/{dp_bayes}", 'rb') as f:
            pattern_model = pickle.load(f)
        pattern_model.n_jobs =1

    if using_quan == 'log':
        parameter += f'_pro({count_prot})_as({add_src})_log({n_components})'
    elif using_quan == 'bayesian':
        parameter += f'_pro({count_prot})_as({add_src})_p0({p0})'
    
    # ip별 퀀타이제이션 셋 만들기
    # if using_quan=='gmm':
    #     if os.path.isfile(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl"):
    #         with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
    #             train_data = pickle.load(f)
                
    #         with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
    #             test_data = pickle.load(f)
                
    #     else:
    #         parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})"

    #         train_data = pattern_gmm.transform_tokenize(train_raw, confidence=confidence)
    #         test_data = pattern_gmm.transform_tokenize(test_raw, confidence=confidence)
    
    if using_quan=='log':
        if os.path.isfile(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl"):
            with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
                train_data = pickle.load(f)
                
            with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
                test_data = pickle.load(f)
                
        else:
            train_data = pattern_model.multi_transform(train_raw)
            test_data = pattern_model.multi_transform(test_raw)

    if using_quan=='bayesian':
        if os.path.isfile(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl"):
            with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
                train_data = pickle.load(f)
                
            with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
                test_data = pickle.load(f)
                
        else:
            train_data = pattern_model.transform(train_raw)
            test_data = pattern_model.transform(test_raw)
    
    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})_vic({add_victim})"

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
        print("src : ", len(train_src))
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
        print("prot : ", len(train_prot))
        train_data = [f"{train}{prt}" for train, prt in zip(train_data, train_prot)]

        test_ffiles_prt = glob.glob(os.path.join(folder, 'test_protflag_*'))
        test_ffiles_prt.sort()
        for file in test_ffiles_prt:
            with open(file, 'rb') as f:
                test_prot += pickle.load(f)

        test_data = [f"{test}{prt}" for test, prt in zip(test_data, test_prot)]

    if using_quan == 'log':
        parameter += f'_pro({count_prot})_as({add_src})_log({n_components})'
    elif using_quan == 'bayesian':
        parameter += f'_pro({count_prot})_as({add_src})_p0({p0})'

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

    # if using_quan == 'gmm':
    #     file_name = f"GMM_as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-n({n_components})-atk({attack})-conf({confidence})_window({train_window}-{test_window})_mm({using_minmax})_vic({add_victim}).csv"
    if using_quan == 'log':
        file_name = f"log({n_components})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax})-vic({add_victim}).csv"
    elif using_quan == 'bayesian':
        file_name = f"p0({p0})_as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax})-vic({add_victim}).csv"

    save_file = f"./result/{dataset_path}/{using_quan}_{file_name}.csv"
    
    if real_time:
        evaluate_realtime(train_multi_dict, train_label, test_data, test_key, save_file)
    else:
        evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file)

if __name__ == "__main__":
    min_data = 5
    change_feature = False
    seperate_attackIP = True
    count_prot = True
    using_minmax = True
    add_src = True
    real_time = 0
    attack = 1 # 0이 정상 1이 공격 2가 혼합
    train_window = 0
    test_window = 10
    victim = True
    p0 = 0.05
    logN = 1.2

    try:
        for data in ['CTU-Rbot', 'CTU-Neris', 'CTU-Virut']:
            for using_quan in ['log', 'bayesian']:
                main(data, min_data, attack, change_feature, add_src, seperate_attackIP, count_prot, train_window, test_window, logN, real_time, using_minmax, using_quan, victim, p0)

    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{add_src} add_src-_attackIP-{seperate_attackIP} test에서 에러 발생\n")
            f.write(f"{error_info}\n\n")
