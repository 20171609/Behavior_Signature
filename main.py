import os
import gc
import csv
import pickle
import glob

import matplotlib.pyplot as plt
from GMM_Quantization import make_gmm
import global_
from utils import *
from profiling import b_profiling
import traceback

def main(dataset_path, attack, change_feature, seperate, change_src, test_method, confidence):
    # dataset_path = "CTU-Rbot"\

    # Profiling에 사용
    min_data = 5
    # change_feature = True
    # change_src = True
    separate_attackIP = False

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

    dp = f"cs({change_src})_cf({change_feature})_sepIP({separate_attackIP})_min({min_data})"
    parameter = '_'.join(dp.split('_')[1:])

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
    for file in train_ffiles:
        with open(file, 'rb') as f:
            train_raw += pickle.load(f)
    
    # 'train_key'로 시작하는 모든 파일 찾기
    train_kfiles = glob.glob(os.path.join(folder, 'train_key*'))
    for file in train_kfiles:
        with open(file, 'rb') as f:
            train_key += pickle.load(f)

    # 'test'로 시작하는 모든 파일 찾기
    test_ffiles = glob.glob(os.path.join(folder, 'test_feature*'))
    for file in test_ffiles:
        with open(file, 'rb') as f:
            test_raw += pickle.load(f)

    with open(f"./{dataset_path}/profiling/{dp}/test_{dp}_key.pkl", 'rb') as f:
        test_key = pickle.load(f)

    if not os.path.isdir(f'./{save_path}'):
        os.mkdir(f'./{save_path}')

    # GMM 생성 부분
    if not os.path.isfile(f"{save_path}/{n_components}n_components_{attack}attack_{global_.change_src}cs.pkl"):
        print("GMM 생성 해야함")
        make_gmm(train_raw, train_key, n_components, save_path)

    with open(f"{save_path}/{n_components}n_components_{attack}attack_{global_.change_src}cs.pkl", 'rb') as f:
        pattern_gmm = pickle.load(f)

    print(f"{n_components} {attack}attack GMM 불러옴")
    # ip별 퀀타이제이션 셋 만들기

    train_data = pattern_gmm.transform_tokenize(train_raw, confidence=confidence)
    test_data = pattern_gmm.transform_tokenize(test_raw, confidence=confidence)

    if change_src:
        with open(f'./{dataset_path}/train_{dp}_srcflag.pkl', 'rb') as f:
            train_src = pickle.load(f)

        train_data = [f"{train}{src}" for train, src in zip(train_data, train_src)]
    
        with open(f'./{dataset_path}/test_{dp}_srcflag.pkl', 'rb') as f:
            test_src = pickle.load(f)

        test_data = [f"{test}{src}" for test, src in zip(test_data, test_src)]
    
    train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set\
          = make_quantization_dict(train_data, train_key)
    
    test_multi_dict, test_single_dict, test_label, _, _\
          = make_quantization_dict(test_data, test_key)

    # evaluate
    print("평가 시작")
    save_file = f"{save_path}/result_n({n_components})-min({min_data})-atk({attack})-cfd({confidence})-cf({change_feature})-cs({change_src})\
        sep({seperate})-test({test_method}).csv"
    evaluate(train_multi_dict, train_single_dict, train_label, attack_quantization_multi_set, attack_quantization_single_set,\
             test_multi_dict, test_single_dict, test_label, save_file)

    #score 측정

if __name__ == "__main__":
    try:
        for data in ['CTU-Rbot', 'CTU-Neris']:
            try:
                for attack in [0, 1, 2]: # 0이 정상 1이 공격 2가 혼합
                    try:
                        for change_feature in [True, False]:
                            try:
                                for seperate in [True, False]:
                                    try:
                                        for seperate_attackIP in [False]:
                                            try:
                                                for change_src in [True, False]:
                                                    try:
                                                        for test_method in [True, False]:
                                                            try:
                                                                for confidence in [1.28, 2.56]:
                                                                    main(data, attack, change_feature, seperate, change_src, test_method, confidence)
                                                            
                                                            except:
                                                                error_info = traceback.format_exc()
                                                                with open('log.txt', 'a') as f:
                                                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                                    f.write(f"{error_info}\n\n")
                                                                continue
                                                    except:
                                                        error_info = traceback.format_exc()
                                                        with open('log.txt', 'a') as f:
                                                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                            f.write(f"{error_info}\n\n")
                                                        continue
                                            except:
                                                error_info = traceback.format_exc()
                                                with open('log.txt', 'a') as f:
                                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                                    f.write(f"{error_info}\n\n")
                                                continue
                                    except:
                                        error_info = traceback.format_exc()
                                        with open('log.txt', 'a') as f:
                                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                            f.write(f"{error_info}\n\n")
                                        continue
                            except:
                                error_info = traceback.format_exc()
                                with open('log.txt', 'a') as f:
                                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                                    f.write(f"{error_info}\n\n")
                                continue
                    except:
                        error_info = traceback.format_exc()
                        with open('log.txt', 'a') as f:
                            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                            f.write(f"{error_info}\n\n")
                        continue
            except:
                error_info = traceback.format_exc()
                with open('log.txt', 'a') as f:
                    f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
                    f.write(f"{error_info}\n\n")
                continue 
    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{seperate} sep-{change_src} change_src-{test_method} sep_attackIP-{seperate_attackIP} test에서 에러 발생\n")
            f.write(f"{error_info}\n\n")   