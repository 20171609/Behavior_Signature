import os
import gc
import csv
import pickle
import glob
from datetime import datetime 

from log_quan import make_log_quan
import global_
from utils import *
from profiling import b_profiling
import traceback
from test import test_live

def main(dataset_path, min_data, attack, change_feature, add_src, count_prot, test_window, n_components, using_minmax, using_entropy, command, n_ip_flow):
    train_path = [rf"dataset\{dataset_path}\train\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'train'))]
    test_attack_path = [rf"dataset\{dataset_path}\test_attack\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'test_attack'))]
    test_benign_path = [rf"dataset\{dataset_path}\test_benign\{file}" for file in os.listdir(os.path.join("./dataset", dataset_path, 'test_benign'))]

    global_.initialize(train_path[0], change_feature, attack, count_prot, test_window, using_minmax,n_ip_flow)

    parameter = f"cf({change_feature})_if({n_ip_flow})_min({min_data})_mm({using_minmax})_c{command}"

    if not os.path.isdir(f"./preprocessing"):
        os.mkdir(f"./preprocessing")

    if not os.path.isdir(f"./preprocessing/{dataset_path}"):
        os.mkdir(f"./preprocessing/{dataset_path}")

    if not os.path.isdir(f'./preprocessing/{dataset_path}/profiling'):
        os.mkdir(f'./preprocessing/{dataset_path}/profiling')

    print("Profiling 시작")
    b_profiling(train_path, "train", parameter, min_data, dataset_path)
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

    if not os.path.isdir(f'./preprocessing/{dataset_path}/LOG'):
        os.mkdir(f'./preprocessing/{dataset_path}/LOG')

    # log datapath
    dp_log = f"entropy({using_entropy})_log_n({n_components})_if({n_ip_flow})_atk({attack})_mm({using_minmax})_cf({change_feature})_min({min_data})_{command}c_log.pkl"
    
    if not os.path.isfile(f"./preprocessing/{dataset_path}/LOG/{dp_log}"):
        print("LOG boundary 생성 해야함")
        make_log_quan(train_raw, train_key, dataset_path, n_components,dp_log,using_entropy)

    print(f"log n:{n_components} {attack}attack LOG 불러옴")
    
    with open(f"./preprocessing/{dataset_path}/LOG/{dp_log}", 'rb') as f:
        pattern_model = pickle.load(f)

    parameter += f'_pro({count_prot})_as({add_src})_log({n_components})'

    train_raw = np.array(train_raw)
    train_data = pattern_model.multi_transform(train_raw)

    parameter = f"cf({change_feature})_if({n_ip_flow})_min({min_data})_mm({using_minmax})_c{command}"
    
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

    print(train_data[0])

    if using_entropy :
        parameter += f'_pro({count_prot})_as({add_src})_log_entropy({n_components})'
    else:
        parameter += f'_pro({count_prot})_as({add_src})_log_bin({n_components})'

    if not os.path.isdir(f"./debug_data"):
        os.mkdir(f"./debug_data")

    if not os.path.isdir(f"./debug_data/{dataset_path}"):
        os.mkdir(f"./debug_data/{dataset_path}")

    if not os.path.isdir(f'./debug_data/{dataset_path}/{parameter}'):
        os.mkdir(f'./debug_data/{dataset_path}/{parameter}')
    
    with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'wb') as f:
        pickle.dump(train_data,f)

    file_name = f"ent{using_entropy}_log({logN})-if({n_ip_flow})-as({add_src})-cf({change_feature})-prot({count_prot})-min({min_data})-atk({attack})-test_window({test_window})-mm({using_minmax})_c{command}.csv"
    save_file = f"./result/{dataset_path}/{file_name}.csv"
    
    print(len(train_data))

    if not os.path.isdir(f'./result'):
        os.mkdir(f'./result')

    if not os.path.isdir(f'./result/{dataset_path}'):
        os.mkdir(f'./result/{dataset_path}')

    # evaluate
    print("평가 시작")

    train_multi_dict, train_label = make_quantization_dict_live_test(train_data, train_key)
    test_live(save_file, test_attack_path, min_data, pattern_model, add_src, train_multi_dict, train_label, benign_test = False)
    test_live(save_file, test_benign_path, min_data, pattern_model, add_src, train_multi_dict, train_label, benign_test = True)
    

if __name__ == "__main__":
    min_data = 10
    change_feature = True
    count_prot = True
    using_minmax = True
    add_src = True
    attack = 1 # 0이 정상 1이 공격 2가 혼합
    
    test_window = 10
    logN = 128
    using_entropy = False
    command = "test2" 
    n_ip_flow = 5000

    try:
        for data in ['test']:
            main(data, min_data, attack, change_feature, add_src, count_prot, test_window, logN, using_minmax, using_entropy, command, n_ip_flow)

    except:
        error_info = traceback.format_exc()
        with open('log.txt', 'a') as f:
            f.write(f"{data}-{attack} attack-{change_feature} changefeature-{add_src} add_src- test에서 에러 발생\n")
            f.write(f"{error_info}\n\n")