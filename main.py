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



def main(dataset_path, attack, change_feature, seperate, change_src, test_method, confidence, separate_attackIP):
    # dataset_path = "CTU-Rbot"\

    # Profiling에 사용
    min_data = 5
    # change_feature = True
    # change_src = True
    # separate_attackIP = False

    # GMM에 사용
    n_components = 20
    # attack = True
    # confidence = 1.28

    # test할 때 사용
    # seperate = True (얜 GMM도)
    # test_method = True

    train_path = [rf"dataset\{dataset_path}\train\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'train'))]
    test_path = [rf"dataset\{dataset_path}\test\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'test'))]

    global_.initialize(train_path[0], change_src, change_feature, seperate, attack, test_method,separate_attackIP)

    parameter = f"cs({change_src})_cf({change_feature})_sepIP({separate_attackIP})_min({min_data})"

    if not os.path.isdir(f"./preprocessing"):
        os.mkdir(f"./preprocessing")

    if not os.path.isdir(f"./preprocessing/{dataset_path}"):
        os.mkdir(f"./preprocessing/{dataset_path}")

    if not os.path.isdir(f'./preprocessing/{dataset_path}/profiling'):
        os.mkdir(f'./preprocessing/{dataset_path}/profiling')

    print("Profiling 시작")
    b_profiling(train_path, "train", parameter, min_data, dataset_path)
    print("Profiling 끝")

    print("test profiling 시작")
    b_profiling(test_path, "test", parameter, min_data, dataset_path)
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

    # GMM 이름
    dp_GMM = f"cs({change_src})_cf({change_feature})_n({n_components})_atk({attack})_conf({confidence})_sep({seperate})_sepIP({separate_attackIP})_GMM.pkl"

    # GMM 생성 부분
    if not os.path.isfile(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}"):
        print("GMM 생성 해야함")
        make_gmm(train_raw, train_key, n_components, dp_GMM, dataset_path)

    print(f"{n_components}n {attack}attack GMM 불러옴")
    with open(f"./preprocessing/{dataset_path}/GMM/{dp_GMM}", 'rb') as f:
        pattern_gmm = pickle.load(f)
    
    # ip별 퀀타이제이션 셋 만들기
    train_data = pattern_gmm.transform_tokenize(train_raw, confidence=confidence)
    test_data = pattern_gmm.transform_tokenize(test_raw, confidence=confidence)

    with open(f"./preprocessing/{dataset_path}/{dataset_path}_{parameter}_train_data.pkl", 'wb') as f:
        pickle.dump(train_data,f)
    with open(f"./preprocessing/{dataset_path}/{dataset_path}_{parameter}_test_data.pkl", 'wb') as f:
        pickle.dump(test_data,f)

    if not change_src:
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

    train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set\
          = make_quantization_dict(train_data, train_key)

    test_multi_dict, test_single_dict, test_label, _, _\
          = make_quantization_dict(test_data, test_key)

    with open(f"./preprocessing/{dataset_path}/{dataset_path}_{parameter}_train_multi_dict.pkl", 'wb') as f:
        pickle.dump(train_multi_dict,f)
    with open(f"./preprocessing/{dataset_path}/{dataset_path}_{parameter}_test_multi_dict.pkl", 'wb') as f:
        pickle.dump(test_multi_dict,f)
    with open(f"./preprocessing/{dataset_path}/{dataset_path}_{parameter}_attack_quantization_multi_set.pkl", 'wb') as f:
        pickle.dump(attack_quantization_multi_set,f)

    if not os.path.isdir(f'./result'):
        os.mkdir(f'./result')

    if not os.path.isdir(f'./result/{dataset_path}'):
        os.mkdir(f'./result/{dataset_path}')

    # evaluate
    print("평가 시작")
    file_name = f"cs({change_src})-cf({change_feature})-sepIP({separate_attackIP})-min({min_data})-n({n_components})-atk({attack})-conf({confidence})-sep({seperate})-test({test_method}).csv"
    save_file = f"./result/{dataset_path}/{file_name}.csv"
    evaluate(train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set,\
             test_multi_dict, test_single_dict, test_label, save_file)

    #score 측정

if __name__ == "__main__":
    try:
        for data in ['CTU-Rbot','CTU-Neris']:
            try:
                for attack in [0]: # 0이 정상 1이 공격 2가 혼합
                    try:
                        for change_feature in [False]:
                            try:
                                for seperate in [False]:
                                    try:
                                        for seperate_attackIP in [True]:
                                            try:
                                                for change_src in [True]:
                                                    try:
                                                        for test_method in [True]: # true가 membership check
                                                            try:
                                                                for confidence in [1.28]:
                                                                    main(data, attack, change_feature, seperate, change_src, test_method, confidence, seperate_attackIP)
                                                            
                                                            except:
                                                                error_info = traceback.format_exc()
                                                                with open('log.txt', 'a') as f:
                                                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                                    now = datetime.now()
                                                                    f.write(f"time: {now}\n")
                                                                    f.write(f"{error_info}\n\n")
                                                                continue
                                                    except:
                                                        error_info = traceback.format_exc()
                                                        with open('log.txt', 'a') as f:
                                                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                            now = datetime.now()
                                                            f.write(f"time: {now}\n")
                                                            f.write(f"{error_info}\n\n")
                                                        continue
                                            except:
                                                error_info = traceback.format_exc()
                                                with open('log.txt', 'a') as f:
                                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                    now = datetime.now()
                                                    f.write(f"time: {now}\n")
                                                    f.write(f"{error_info}\n\n")
                                                continue
                                    except:
                                        error_info = traceback.format_exc()
                                        with open('log.txt', 'a') as f:
                                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                            now = datetime.now()
                                            f.write(f"time: {now}\n")
                                            f.write(f"{error_info}\n\n")
                                        continue
                            except:
                                error_info = traceback.format_exc()
                                with open('log.txt', 'a') as f:
                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                    now = datetime.now()
                                    f.write(f"time: {now}\n")
                                    f.write(f"{error_info}\n\n")
                                continue
                    except:
                        error_info = traceback.format_exc()
                        with open('log.txt', 'a') as f:
                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                            now = datetime.now()
                            f.write(f"time: {now}\n")
                            f.write(f"{error_info}\n\n")
                        continue
            except:
                error_info = traceback.format_exc()
                with open('log.txt', 'a') as f:
                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                    now = datetime.now()
                    f.write(f"time: {now}\n")
                    f.write(f"{error_info}\n\n")
                continue 
    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
            now = datetime.now()
            f.write(f"time: {now}\n")
            f.write(f"{error_info}\n\n")   