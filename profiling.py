from utils import *
import global_
from tqdm.auto import tqdm
import pickle
import os

class Profile:

    attr_list = ['target_ip', 'target_port', 'opposite_ip', 'opposite_port', 'duration',
                 'target_pkts', 'opposite_pkts', 'target_bytes', 'opposite_bytes',
                'start_time', 'end_time']

    attr_typing_map = {
        'target_ip': lambda x: str(x),
        'target_port': lambda x: int(x, 16) if x.startswith('0x') else int(float(x)),
        'opposite_ip': lambda x: str(x),
        'opposite_port': lambda x: int(x, 16) if x.startswith('0x') else int(float(x)),
        'duration': lambda x: float(x),
        'target_pkts': lambda x: int(float(x)),
        'opposite_pkts': lambda x: int(float(x)),
        'target_bytes': lambda x: int(float(x)),
        'opposite_bytes': lambda x: int(float(x)),
        'start_time' : lambda x: str(x),
        'end_time' : lambda x: str(x)
    }

    def __init__(self, profile_key):
        self.__profile_key = profile_key
        self.__table = {}
        for attr in self.attr_list:
            self.__table[attr] = []

    def add(self, attr_dict: dict):
        for attr in self.attr_list:
            value = self.attr_typing_map[attr](attr_dict[attr])
            self.__table[attr].append(value)

    def debug(self):
        ret_str = self.__str_attr() + '\n'
        for idx in range(self.__flow_cnt):
            ret_str += self.__str_row(idx) + '\n'
        return ret_str

    def __str_attr(self) -> str:
        return ",".join(self.attr_list)

    def __str_row(self, idx: int) -> str:
        print_row_values = []
        for attr in self.attr_list:
            print_row_values.append(str(self.__table[attr][idx]))
        return ",".join(print_row_values)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, attr) -> list:
        return self.table[attr]

    @property
    def table(self) -> dict:
        return self.__table

    @property
    def profile_key(self) -> str:
        return self.__profile_key

    @property
    def target_ip(self) -> str:
        return self.profile_key.split('_')[0]
    

def profiling(flow_list, target_ip, st_time, end_time):
    profile_key = '{}_{}_{}'.format(target_ip, get_str_time(st_time), get_str_time(end_time))
    new_pf = Profile(profile_key)
    for flow in flow_list:
        new_pf.add(add_flow(flow, target_ip))
    return (new_pf, profile_key)

def add_flow(flow: list, target_ip):
    target_ip = target_ip.split('_')[0]
    sip = flow[global_.column_index['source']]

    attr_map = {}
    if target_ip == sip:
        attr_map = global_.attribute_map
    else:
        attr_map = global_.attribute_map_inv
    attr_dict = {}
    for attr, column in attr_map.items():
        attr_dict[attr] = flow[global_.column_index[column]]
    return attr_dict

## Behavior Profiling
def b_profiling(data_path, save_path, min_data):
    profile_list = []
    profile_key_list = []
    profile_srcflag = []

    for file in data_path:
        flow_stack = {}
        print(file)
        file_name = file.split('\\')[3].split('.')[0]
        print(file_name)

        with open(file, 'r', encoding='utf-8') as f:
            col = f.readline().strip().split(',')
            column_index = {i : idx for idx, i in enumerate(col)}
            csv_data = f.readlines()

            for idx, tmp_flow in enumerate(tqdm(csv_data)):
                flow = tmp_flow.strip().split(',')
                if flow[0] == '':
                    continue

                flow[column_index['first']] = flow[column_index['first']].split('.')[0].replace('/','-')
                    
                if flow[column_index['src_port']] == '':
                    flow[column_index['src_port']] = "-1"

                if flow[column_index['dst_port']] == '':
                    flow[column_index['dst_port']] = "-1"
                
                sip, dip = flow[column_index['source']], flow[column_index['destination']]
                now_time = get_int_time(flow[column_index['first']])

                ip_set_list = [sip, dip, f"{sip}_{dip}", f"{dip}_{sip}"] if global_.seperate else [sip, dip]

                for target_ip in ip_set_list:
                    if target_ip not in flow_stack:
                        flow_stack[target_ip] = {'flow':[], 'label':[],  'srcflag' : []}
                        flow_stack[target_ip]['st_time'] = now_time

                    if "*" in target_ip.split('_')[0]:
                        flow_stack[target_ip]['label'].append("Attack")
                    else:
                        flow_stack[target_ip]['label'].append('Benign')

                    if target_ip.split('_')[0] == sip:
                        flow_stack[target_ip]['srcflag'].append(1)
                    else:
                        flow_stack[target_ip]['srcflag'].append(0)

                    flow_stack[target_ip]['flow'].append(flow)
                    flow_stack[target_ip]['end_time'] = now_time

                    if len(flow_stack[target_ip]['flow']) == min_data:
                        profile, profile_key = profiling(flow_stack[target_ip]['flow'], target_ip, flow_stack[target_ip]['st_time'], flow_stack[target_ip]['end_time'])
                        profile_list.append(profile)
                        profile_key_list.append(f"{check_label(flow_stack[target_ip]['label'])}+{profile_key}+{file_name}")
                        profile_srcflag.append(sum(flow_stack[target_ip]['srcflag']))
                        flow_stack[target_ip]['flow'].pop(0)
                        flow_stack[target_ip]['label'].pop(0)
                        flow_stack[target_ip]['srcflag'].pop(0)
                        flow_stack[target_ip]['st_time'] = get_int_time(flow_stack[target_ip]['flow'][0][column_index['first']])

    feature_func_map = global_.feature_func_map
    feature_list = list(feature_func_map.keys())                
        
    feature_matrix = [[] for _ in range(len(feature_list))]

    for profile in tqdm(profile_list):
        for i, feature in enumerate(feature_list):
            feature_matrix[i].append(feature_func_map[feature](profile))

    feature = np.array(feature_matrix).T.tolist()
    
    sf = save_path.split('_')[-1]

    if not os.path.isdir(f'./{sf}'):
        os.mkdir(f'./{sf}')

    with open(f'./{sf}/{save_path}_feature.pkl', 'wb') as f:
        pickle.dump(feature, f)

    with open(f'./{sf}/{save_path}_key.pkl', 'wb') as f:
        pickle.dump(profile_key_list, f)

    with open(f'./{sf}/{save_path}_srcflag.pkl', 'wb') as f:
        pickle.dump(profile_srcflag, f)