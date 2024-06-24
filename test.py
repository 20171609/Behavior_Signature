from profiling import Profile
import os
import global_
from tqdm.auto import tqdm
from collections import Counter, deque, defaultdict
from utils import *
import csv
from profiling import b_profiling
import glob
import pickle
import gc
import pandas as pd

def profiling(flow_list, target_ip):
    profile_key = target_ip
    new_pf = Profile(profile_key)
    for flow in flow_list:
        new_pf.add(add_flow(flow, target_ip))
    return (new_pf, profile_key)

def add_flow(flow: list, target_ip):
    target_ip = target_ip.split('_')[0]
    sip = flow['source'] #flow[global_.column_index['source']]
    
    attr_map = {}
    if target_ip == sip:
        attr_map = global_.attribute_map
    else:
        attr_map = global_.attribute_map_inv
    attr_dict = {}
    for attr, column in attr_map.items():
        attr_dict[attr] = flow[column]#global_.column_index[column]]
    return attr_dict

def make_remain_label(tmp):
    tmp = set(tmp)

    if len(tmp) == 1:
        return tmp.pop()

    else:
        for t in tmp:
            if t.upper() != 'BENIGN':
                return t

def find_label(label_dict, ip_list):
    label_set = set()
    
    for ip in ip_list:
        for label in label_dict[ip]:
            if label != 'BENIGN':
                label_set.add(label)

    if len(label_set) == 0:
        label_set.add('BENIGN')

    return label_set

def test_live(save_path, data_path, min_data, ignore_background, log, add_src, train_dict, train_label, benign_test):
    feature_func_map = global_.feature_func_map
    feature_list = list(feature_func_map.keys())

    for file in data_path:
        pred_dict = defaultdict(set)
        score_dict = dict()
        compare_dict = defaultdict(dict)
        sequence = defaultdict(deque)
        num_signature = defaultdict(dict)
        label_dict = defaultdict(set)
        max_train_ip = defaultdict(set)
        done_test_ip = set()
        # 실시간 프로파일링
        # ip별로 10개가 차면 해당 profile을 퀀타이제이션 하기
        # 퀀타이제이션 된 값을 이용해서 각 train IP별 유사도 계산하여 유사도 증가시키기
        global_.change_col(file)

        flow_stack = {}
        print(file)
        file_name = file.split('\\')[-1].split('.')[0]
        print(file_name)

        df = pd.read_csv(file)
        df['src_port'].fillna("-1", inplace=True)
        df['dst_port'].fillna("-1", inplace=True)

        for idx, flow in tqdm(df.iterrows(), total=len(df)):
            if flow['source'] == '':
                continue

            if ignore_background:
                if flow['Label'].upper() == 'BACKGROUND':
                    continue

            # if flow['src_port'] == '':
            #     flow['src_port'] = "-1"

            # if flow['dst_port'] == '':
            #     flow['dst_port'] = "-1"

            sip, dip = flow['source'], flow['destination']
            
            for target_ip in [sip, dip]:
                if benign_test:
                    if '*' in target_ip:
                        continue
                if target_ip in done_test_ip:
                    continue
                if target_ip not in score_dict:
                    score_dict[target_ip] = 0
                    
                if target_ip not in flow_stack:
                    flow_stack[target_ip] = {'flow': deque([]), 'label':deque([]),  'srcflag' : deque([]), 'protCount' : deque([]), 'total_src' : 0}


                if "*" in target_ip.split('_')[0]:
                    flow_stack[target_ip]['label'].append(flow['Label'].upper())
                    label_dict[target_ip].add(flow['Label'].upper())
                else:
                    if 'BENIGN' not in flow['Label'].upper() and 'BACKGROUND' not in flow['Label'].upper():
                        flow_stack[target_ip]['label'].append('BENIGN')
                        label_dict[target_ip].add('BENIGN')
                    
                    else:
                        flow_stack[target_ip]['label'].append(flow['Label'].upper())
                        label_dict[target_ip].add(flow['Label'].upper())

                if target_ip.split('_')[0] == sip:
                    flow_stack[target_ip]['srcflag'].append(1)
                    flow_stack[target_ip]['total_src'] += 1
                else:
                    flow_stack[target_ip]['srcflag'].append(0)
                    flow_stack[target_ip]['total_src'] += 0

                if global_.count_prot:
                    flow_stack[target_ip]['protCount'].append(flow['prot'])
                
                flow_stack[target_ip]['flow'].append(flow)

                if len(flow_stack[target_ip]['flow']) == min_data:
                    profile, profile_key = profiling(flow_stack[target_ip]['flow'], target_ip)

                    tmp = []

                    for i, feature in enumerate(feature_list):
                        tmp.append(feature_func_map[feature](profile))
                    
                    # 표준편차 제거
                    # for i in range(8, 13):
        

                    if global_.count_prot:
                        count_tmp = [0, 0, 0] # tcp, udp, icmp

                        for p in flow_stack[target_ip]['protCount']:
                            p = str(p)
                            if p.upper() == 'TCP' or p == '6':
                                count_tmp[0] = 1
                            
                            elif p.upper() == 'UDP' or p == '17':
                                count_tmp[1] = 1
                            
                            elif p.upper() == 'ICMP' or p == '1':
                                count_tmp[2] = 1

                    tmp = log.multi_transform([tmp], False)
                    tmp = tmp[0]
                    # 여기서 퀀타이제이션 만들기
                    if add_src:
                        tmp = f"{tmp}{flow_stack[target_ip]['total_src']}"

                    if global_.count_prot:
                        tmp = f"{tmp}{count_tmp}"

                    sequence[target_ip].append(tmp)

                    if tmp not in num_signature[target_ip]:
                        num_signature[target_ip][tmp] = 0
                    num_signature[target_ip][tmp] += 1
                    
                    now = ''
                    if len(sequence[target_ip]) > global_.test_window:
                        now = sequence[target_ip].popleft()
                        num_signature[target_ip][now] -= 1

                    # 여기서 해당 IP의 유사도 구하기.
                    # train IP 마다
                    # 새로 생긴 tmp가 있으면 +1
                    # 나온 signature가 있으면 -1
                    # 개수 반영 해야함.
                    # max값 갱신되면 해당 train ip와 점수 넣기
                    if now == tmp:
                        flow_stack[target_ip]['total_src'] -= flow_stack[target_ip]['srcflag'].popleft()
                        
                        if global_.count_prot:
                            flow_stack[target_ip]['protCount'].popleft()
                        
                        flow_stack[target_ip]['flow'].popleft()
                        flow_stack[target_ip]['label'].popleft()
                        continue

                    for train_ip, signatures in train_dict.items():
                        if train_ip in max_train_ip[target_ip]:
                            continue
                        
                        if train_ip not in compare_dict[target_ip]:
                            compare_dict[target_ip][train_ip] = 0
                    
                        if tmp in signatures:
                            if num_signature[target_ip][tmp] <= signatures[tmp]:
                                compare_dict[target_ip][train_ip] += 1
                        
                        if now in signatures:
                            if num_signature[target_ip][now] < signatures[now]:
                                compare_dict[target_ip][train_ip] -= 1

                        if compare_dict[target_ip][train_ip] > score_dict[target_ip]:
                            score_dict[target_ip] = compare_dict[target_ip][train_ip]
                            pred_dict[target_ip] = set([train_ip])
                        
                        elif compare_dict[target_ip][train_ip] == score_dict[target_ip]:
                            pred_dict[target_ip].add(train_ip)
                        
                        if compare_dict[target_ip][train_ip] == global_.test_window:
                            max_train_ip[target_ip].add(train_ip)
                            test_label = find_label(label_dict, [target_ip])
                            train_label_set = find_label(train_label, pred_dict[target_ip])
                            for label1 in test_label:
                                if label1 == 'BENIGN':
                                    continue
                                else:
                                    if label1 in train_label_set:
                                        done_test_ip.add(target_ip)
                    
                    flow_stack[target_ip]['total_src'] -= flow_stack[target_ip]['srcflag'].popleft()
                    
                    if global_.count_prot:
                        flow_stack[target_ip]['protCount'].popleft()
                    
                    flow_stack[target_ip]['flow'].popleft()
                    flow_stack[target_ip]['label'].popleft()

        # profile이 생성되지 않은 데이터에 대해서 채점하기 위한 코드
        remain_ip_set = set(flow_stack.keys()) - set(sequence.keys())

        file_exists = os.path.isfile(save_path) and os.path.getsize(save_path) > 0
        # csv 적을 때 test IP에 file name 넣기
        with open(f"{save_path}", "a", newline='', encoding='utf-8') as f:
            wr = csv.writer(f)

            if not file_exists:
                wr.writerow(["Test IP", "Test IP Label", "Max IP", "Max IP Label", "Max Sim"])

            for test_ip in sequence.keys():
                if score_dict[test_ip] > 0:
                    test_label = find_label(label_dict, [test_ip])
                    train_label_set = find_label(train_label, pred_dict[test_ip])
                    wr.writerow([f"{test_ip}_{file_name}", test_label.pop(), pred_dict[test_ip], train_label_set, score_dict[test_ip] / global_.test_window])
                else:
                    test_label = find_label(label_dict, [test_ip])
                    wr.writerow([f"{test_ip}_{file_name}", test_label.pop(), '-', 'BENIGN', 0])
                ## train 부분 시그니처 Counter로 만들기
                ## train 라벨 가져오는 코드 작성하기
                ## test 라벨 생성하는 함수로 작성하기

            for remain_ip in remain_ip_set:
                if '*' in remain_ip:
                    wr.writerow([f"{remain_ip}_{file_name}", make_remain_label(flow_stack[remain_ip]['label']), '-', 'BENIGN', -1])

        del pred_dict
        del score_dict
        del compare_dict
        del sequence
        del num_signature
        del label_dict
        del max_train_ip
        del df


def test_no_live(save_file, test_path, parameter, min_data, dataset_path, ignore_background, log, add_src, count_prot, attack, train_multi_dict,  train_label, attack_quantization_multi_set):
    print("test profiling 시작")
    b_profiling(test_path, "test", parameter, min_data, dataset_path, ignore_background)
    print("Test 끝")

    test_raw = []
    test_key = []

    folder = f'./preprocessing/{dataset_path}/profiling/{parameter}'

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

    test_data = log.multi_transform(test_raw)
    test_src = []
    test_prot = []

    if add_src:
        test_ffiles_src = glob.glob(os.path.join(folder, 'test_srcflag*'))
        test_ffiles_src.sort()
        for file in test_ffiles_src:
            with open(file, 'rb') as f:
                test_src += pickle.load(f)

        test_data = [f"{test}{src}" for test, src in zip(test_data, test_src)]

    if count_prot:
        test_ffiles_prt = glob.glob(os.path.join(folder, 'test_protflag_*'))
        test_ffiles_prt.sort()
        for file in test_ffiles_prt:
            with open(file, 'rb') as f:
                test_prot += pickle.load(f)

        test_data = [f"{test}{prt}" for test, prt in zip(test_data, test_prot)]


    #         with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
    #             test_data = pickle.load(f)

    # with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'wb') as f:
    #     pickle.dump(test_data,f)
    
    test_multi_dict, test_label = make_quantization_test_dict(test_data, test_key)
    
    # with open(f"./debug_data/{dataset_path}/{parameter}/test_multi_dict_attack{attack}.pkl", 'wb') as f:
    #     pickle.dump(test_multi_dict,f)

    evaluate_original(train_multi_dict,  train_label, attack_quantization_multi_set, test_multi_dict, test_label, save_file)