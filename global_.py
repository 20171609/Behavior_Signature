import numpy as np

def initialize(path, change_feature_, attack_, count_prot_, test_window_, using_minmax,n_max_flow_):
    global column_index
    
    with open(path, 'r', encoding='utf-8') as f:
        col = f.readline().strip().split(',')
        column_index = {i : idx for idx, i in enumerate(col)}

    print(column_index)
    
    global attribute_map
    global attribute_map_inv
    global feature_func_map
    global attack

    global change_feature
    global count_prot
    global test_window
    global n_max_flow

    change_feature = change_feature_
    count_prot = count_prot_
    test_window = test_window_
    n_max_flow = n_max_flow_ 
    attack = attack_
    
    attribute_map = {'target_ip': 'source', 'target_port': 'src_port', 'opposite_ip': 'destination', 'opposite_port': 'dst_port',
                'duration': 'duration', 'target_pkts': 'out_pkts', 'opposite_pkts': 'in_pkts',
                'target_bytes': 'out_bytes', 'opposite_bytes': 'in_bytes'}
    
    attribute_map_inv = {'target_ip': 'destination', 'target_port': 'dst_port', 'opposite_ip': 'source', 
                        'opposite_port': 'src_port', 'duration': 'duration', 'target_pkts': 'in_pkts', 
                        'opposite_pkts': 'out_pkts', 'target_bytes': 'in_bytes', 'opposite_bytes': 'out_bytes'} 

    feature_func_map = {
        'card_target_port':
            lambda x: len(set(x['target_port'])),
        'card_opposite_ip':
            lambda x: len(set(x['opposite_ip'])),
        'card_opposite_port':
            lambda x: len(set(x['opposite_port'])),
        'avg_target_pkts':
            lambda x: np.mean(x['target_pkts']),
        'avg_opposite_pkts':
            lambda x: np.mean(x['opposite_pkts']),
        'avg_target_bytes':
            lambda x: np.mean(x['target_bytes']),
        'avg_opposite_bytes':
            lambda x: np.mean(x['opposite_bytes']),
        'avg_dur':
            lambda x: np.mean(x['duration']),
        'std_target_pkts':
            lambda x: np.std(x['target_pkts']) if not change_feature else np.std(x['target_pkts']) + np.mean(x['target_pkts']),
        'std_opposite_pkts':
            lambda x: np.std(x['opposite_pkts']) if not change_feature else np.std(x['opposite_pkts']) + np.mean(x['opposite_pkts']),
        'std_target_bytes':
            lambda x: np.std(x['target_bytes']) if not change_feature else np.std(x['target_bytes']) + np.mean(x['target_bytes']),
        'std_opposite_bytes':
            lambda x: np.std(x['opposite_bytes']) if not change_feature else np.std(x['opposite_bytes']) + np.mean(x['opposite_bytes']),
        'std_dur':
            lambda x: np.std(x['duration']) if not change_feature else np.std(x['duration']) + np.mean(x['duration']),
        'max_target_pkts':
            lambda x: np.max(x['target_pkts']),
        'max_opposite_pkts':
            lambda x: np.max(x['opposite_pkts']),
        'max_target_bytes':
            lambda x: np.max(x['target_bytes']),
        'max_opposite_bytes':
            lambda x: np.max(x['opposite_bytes']),
        'max_duration':
            lambda x: np.max(x['duration']),
        'min_target_pkts':
            lambda x: np.min(x['target_pkts']),
        'min_opposite_pkts':
            lambda x: np.min(x['opposite_pkts']),
        'min_target_bytes':
            lambda x: np.min(x['target_bytes']),
        'min_opposite_bytes':
            lambda x: np.min(x['opposite_bytes']),
        'min_duration':
            lambda x: np.min(x['duration'])
    } if using_minmax else \
    {
        'card_target_port':
            lambda x: len(set(x['target_port'])),
        'card_opposite_ip':
            lambda x: len(set(x['opposite_ip'])),
        'card_opposite_port':
            lambda x: len(set(x['opposite_port'])),
        'avg_target_pkts':
            lambda x: np.mean(x['target_pkts']),
        'avg_opposite_pkts':
            lambda x: np.mean(x['opposite_pkts']),
        'avg_target_bytes':
            lambda x: np.mean(x['target_bytes']),
        'avg_opposite_bytes':
            lambda x: np.mean(x['opposite_bytes']),
        'avg_dur':
            lambda x: np.mean(x['duration']),
        'std_target_pkts':
            lambda x: np.std(x['target_pkts']) if not change_feature else np.std(x['target_pkts']) + np.mean(x['target_pkts']),
        'std_opposite_pkts':
            lambda x: np.std(x['opposite_pkts']) if not change_feature else np.std(x['opposite_pkts']) + np.mean(x['opposite_pkts']),
        'std_target_bytes':
            lambda x: np.std(x['target_bytes']) if not change_feature else np.std(x['target_bytes']) + np.mean(x['target_bytes']),
        'std_opposite_bytes':
            lambda x: np.std(x['opposite_bytes']) if not change_feature else np.std(x['opposite_bytes']) + np.mean(x['opposite_bytes']),
        'std_dur':
            lambda x: np.std(x['duration']) if not change_feature else np.std(x['duration']) + np.mean(x['duration'])
    }

def change_col(path):
    global column_index
    
    with open(path, 'r', encoding='utf-8') as f:
        col = f.readline().strip().split(',')
        column_index = {i : idx for idx, i in enumerate(col)}