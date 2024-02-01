import os
import pandas as pd
import pickle
from tqdm.auto import tqdm
import json
import glob

class IP_debug():
    def __init__(self, ip):
        global train_dataset
        global test_dataset
        global train_multi
        global test_multi
        global train_data
        global test_data
        global train_raw
        global test_raw
        global train_key
        global test_key
        global min_data
        
        self.type = 'train' if 'train' in ip else 'test' if 'test' in ip else 'none'
        self.ip = ip.split('_')[1] if self.type == 'train' else ip.split('_')[0]
        self.idx_quan = dict()
        print(self.ip)
        self.file_name = '_'.join(ip.split('_')[2:]) if self.type == 'train' else '_'.join(ip.split('_')[1:])
        check = train_dataset if self.type == 'train' else test_dataset if self.type == 'test' else dict()

        tmp_df = check[self.file_name]
        tmp_df['source'] = tmp_df['source'].str.replace(r'*', '')
        tmp_df['destination'] = tmp_df['destination'].str.replace(r'*', '')
        self.flow_data = tmp_df[(tmp_df['source'] == self.ip) | (tmp_df['destination'] == self.ip)]

        self.flow_data.fillna(-1, inplace=True)
        
        if self.type == 'train':
            self.multi_set = set(train_multi[ip]) if type(train_multi[ip]) == list else train_multi[ip]
        elif self.type == 'test':
            self.multi_set = set(test_multi[ip]) if type(test_multi[ip]) == list else test_multi[ip]
        else:
            self.multi_set = set()
        
        self.quan_index = dict()
        self.quan_label = dict()
        self.idx_index = dict()
        
        type_data = train_data if self.type == 'train' else test_data if self.type == 'test' else []
        type_key = train_key if self.type == 'train' else test_key if self.type == 'test'else []
        
        now = 0
        
        for idx, key in enumerate(type_key):
            label, tmp_key, file = key.split('+')
            
            if self.ip == tmp_key and self.file_name == file.split('.')[0]:
                quan = type_data[idx]
                
                if quan not in self.quan_index:
                    self.quan_index[quan] = []
                    self.quan_label[quan] = []
                
                self.idx_quan[now] = quan
                self.quan_index[quan].append(now)
                self.quan_label[quan].append(label)
                self.idx_index[now] = idx
                now += 1
        
        self.total_len = now
    
    def get_idx_quan(self, idx):
        return self.idx_quan[idx]
    
    def get_multi_set(self):
        return self.multi_set

    def get_quan_index(self, quan):
        return self.quan_index[quan]

    def get_quan_label(self, quan):
        return self.quan_label[quan]
    
    def find_profile(self, quan):
        idea_bank = set()
        
        index_list = self.quan_index[quan]
        type_raw = train_raw if self.type == 'train' else test_raw if self.type == 'test' else []
        
        for idx in index_list:
            idea_bank.add(type_raw[self.idx_index[idx]])
        
        return idea_bank
    
    def find_profile_idx(self, idx):
        type_raw = train_raw if self.type == 'train' else test_raw if self.type == 'test' else []
        tmp = type_raw[self.idx_index[idx]]
        result = []
        for i in tmp[:13]:
            result.append(round(i, 5))
        return result
    
    def get_flow_idx(self, idx):
        global min_data
        selected_data = self.flow_data.iloc[idx:idx+min_data]
        
        print(idx, len(self.flow_data))
        
        total_data = []
        check = False
        if 'first' in self.flow_data.columns:
            check = True
        for _, data in selected_data.iterrows():
            for idx, d in enumerate(data):
                if check:
                    if idx == 5 or idx == 6:
                        continue
                total_data.append(d)

        return total_data
    
    def generate_html(self):
        # 미리 모든 find_profile_idx 결과를 계산
        profile_data = {idx: self.find_profile_idx(idx) for idx in self.idx_index.keys()}
        
        flow_data = {idx: self.get_flow_idx(idx) for idx in self.idx_index.keys()}
        ip_ = self.ip
        
        html_content = '<html><head>'
        html_content += '<script>'
        html_content += 'var profileData = ' + str(profile_data).replace("'", '"') + ';\n'
        html_content += 'var flow_data = ' + str(flow_data).replace("'", '"') + ';\n'
        html_content += 'var ip_ = ' + json.dumps(ip_) + ';\n'
        html_content += ''' 
        
        function toggleData(quan) {
            var dataDiv = document.getElementById('data-' + quan);
            console.log(dataDiv)
            if (dataDiv) {
                dataDiv.style.display = dataDiv.style.display === 'none' ? 'block' : 'none';
            }
        }
        function showModal(idx, event) {
            var modalId = 'modal';
            var modalContentId = 'modal-content';
            var modalTitle = 'modal-title';
            var modal = document.getElementById(modalId);
            var modalContent = document.getElementById(modalContentId);
            var modaltitleContent = document.getElementById(modalTitle);
            var flowData = flow_data[idx];
            // 모달 제목 추가
            var title = 'ip: ' + ip_;
            
            console.log(profileData);
            console.log(idx);
            
            // profileData[idx]를 테이블로 변환
            var data = profileData[idx];
            var tableHtml = '<table border="2"><tr>';

            // 테이블 헤더 생성
            var header = ['TPort Num', 'OIP Num', 'Oport Num', 'T Avg Pkt', 'O Avg Pkt', 'T Avg Bytes', 'O Avg Bytes', 'Avg_Dur', 'S Std Pkt', 'O Std Pkt', 'S Std Bytes', 'O Std Bytes', 'std_dur']
            for (var key in header) {
                tableHtml += '<th>' + header[key] + '</th>';
            }
            tableHtml += '</tr>';
            
            var data = profileData[idx];
            let count = 0;
            
            tableHtml += '<tr>';
            
            for (var key in data) {
                tableHtml += '<td>' + data[key] + '</td>';
            }
            tableHtml += '</tr>';
            tableHtml += '</table>';
            
            var flow_header = ['SIP', 'DIP', 'proto', 'Sport', 'Dport', 'Inpkt', 'Outpkt', 'Inbytes', 'Outbytes', 'Duration', 'Label']
            tableHtml += '<table border="2"><tr>';
            
            for (var key in flow_header) {
                tableHtml += '<th>' + flow_header[key] + '</th>';
            }
            tableHtml += '</tr>';
            tableHtml += '<tr>';
            
            for (var key in flowData) {
                tableHtml += '<td>' + flowData[key] + '</td>';
                count += 1;

                // 11개의 열마다 새로운 행을 시작
                if (count % 11 === 0) {
                    tableHtml += '</tr><tr>';
                }
            }
            
            tableHtml += '</tr>';
            tableHtml += '</table>';
            

            modalContent.innerHTML = tableHtml;
            modaltitleContent.innerHTML = '<h3>' + title + '</h3>';
            modal.style.display = 'block';

            // 모달 위치 설정
            var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            var windowWidth = window.innerWidth;
            var modalWidth = modal.offsetWidth;

            modal.style.top = (event.clientY + scrollTop) + 'px';


            modal.style.left = '0px';
        }
        function closeModal() {
            var modalId = 'modal';
            var modal = document.getElementById(modalId);
            modal.style.display = 'none';
        }
        </script>
        <style>
        .row {
            display: flex;
        }
        .column {
            flex: 50%;
            padding: 20px;
            box-sizing: border-box;
        }
        .modal {
            display: none;
            position: fixed; /* fixed로 변경 */
            z-index: 1;
            left: 25%; /* 화면의 25% 지점에 위치 */
            top: 25%; /* 화면의 상단에서 25% 지점에 위치 */
            width: 80% /* 너비를 화면의 50%로 설정 */
            height: auto; /* 높이를 자동으로 설정 */
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .button-benign {
            background-color: blue;
            color: white;
        }
        .button-attack {
            background-color: red;
            color: white;
        }
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 100%; /* 내용을 모달의 너비에 맞춤 */
        }
        table {
            border-collapse: collapse;
            width: 100%; /* 테이블을 모달의 너비에 맞춤 */
        }

        th, td {
            border: 1px solid black;
            text-align: left;
            padding: 8px;
        }
        td {
            background-color : white;
        }

        th {
            background-color: #f2f2f2;
        }
        
        </style>
        </head><body>'''
        for quan in self.multi_set:
            # Train과 Test의 라벨 세트를 확인
            labels = self.get_quan_label(quan)
            
            # 라벨에 따른 클래스 결정
            label_class = "button-attack" if "Attack" in labels else "button-benign"
            html_content += '<div class="row">'
            # Train Column
            html_content += '<div class="column">'
            html_content += f'<button onclick="toggleData(\'{quan}\')">{quan}</button><br>'
            html_content += f'<div id="data-{quan}" style="display:none;">'
            for idx, i in enumerate(self.get_quan_index(quan)):
                _class = "button-attack" if labels[idx].upper() != 'BENIGN' else "button-benign"
                html_content += f'<button class="{_class}" onclick="showModal({i},  event)">{i}</button>'
            html_content += '</div>'
            html_content += '</div>'

        # Train 모달
        html_content += '''
        <div id="modal" class="modal">
            <div class="modal-content">
                <span onclick="closeModal()" class="close">&times;</span>
                <h3 id="modal-title"></h3>
                <p id="modal-content"></p>
            </div>
        </div>'''

        return html_content

    def save_html(self, filename):
        html_content = self.generate_html()
        with open(filename, 'w') as file:
            file.write(html_content)
    
class IP_compare():
    def __init__(self, ip1, ip2):
        global min_data
        self.train_ip = IP_debug(ip1)
        self.test_ip = IP_debug(ip2)
        
        self.intersection = self.train_ip.get_multi_set().intersection(self.test_ip.get_multi_set())
        self.attack_inter = set([i for i in self.intersection])
        self.normal_inter = self.intersection - self.attack_inter
        
    def generate_html(self):
        # 미리 모든 find_profile_idx 결과를 계산
        profile_data_train = {idx: self.train_ip.find_profile_idx(idx) for idx in self.train_ip.idx_index.keys() if self.train_ip.idx_quan[idx] in self.intersection}
        profile_data_test = {idx: self.test_ip.find_profile_idx(idx) for idx in self.test_ip.idx_index.keys() if self.test_ip.idx_quan[idx] in self.intersection}
        
        flow_data_train = {idx: self.train_ip.get_flow_idx(idx) for idx in self.train_ip.idx_index.keys() if self.train_ip.idx_quan[idx] in self.intersection}
        flow_data_test = {idx: self.test_ip.get_flow_idx(idx) for idx in self.test_ip.idx_index.keys() if self.test_ip.idx_quan[idx] in self.intersection}
        
        train_ip_ = self.train_ip.ip
        test_ip_ = self.test_ip.ip
        
        html_content = '<html><head>'
        html_content += '<script>'
        html_content += 'var profileDataTrain = ' + str(profile_data_train).replace("'", '"') + ';\n'
        html_content += 'var profileDataTest = ' + str(profile_data_test).replace("'", '"') + ';\n'
        html_content += 'var flow_data_train = ' + str(flow_data_train).replace("'", '"') + ';\n'
        html_content += 'var flow_data_test = ' + str(flow_data_test).replace("'", '"') + ';\n'
        html_content += 'var Total_len_train = ' + str(len(self.train_ip.get_multi_set())).replace("'", '"') + ';\n'
        html_content += 'var Total_len_test = ' + str(len(self.test_ip.get_multi_set())).replace("'", '"') + ';\n'
        html_content += 'var train_ip = ' + json.dumps(train_ip_) + ';\n'
        html_content += 'var test_ip = ' + json.dumps(test_ip_) + ';\n'
        html_content += ''' 
        
        function toggleData(quan, type) {
            var dataDiv = document.getElementById('data-' + type + '-' + quan);
            if (dataDiv) {
                dataDiv.style.display = dataDiv.style.display === 'none' ? 'block' : 'none';
            }
        }
        function showModal(idx, type, event) {
            var modalId = type === 'train' ? 'modal-train' : 'modal-test';
            var modalContentId = type === 'train' ? 'modal-content-train' : 'modal-content-test';
            var modalTitle = type === 'train' ? 'modal-title-train' : 'modal-title-test';
            var modal = document.getElementById(modalId);
            var modalContent = document.getElementById(modalContentId);
            var modaltitleContent = document.getElementById(modalTitle);
            var profileData = type === 'train' ? profileDataTrain : profileDataTest;
            var flowData = type === 'train' ? flow_data_train[idx] : flow_data_test[idx];
            var totalLen = type === 'train' ? Total_len_train : Total_len_test;
            // 모달 제목 추가
            var title = type === 'train' ? type + ' ip : ' + train_ip : type + ' ip: ' + test_ip;
            
            console.log(title)

            // profileData[idx]를 테이블로 변환
            var data = profileData[idx];
            var tableHtml = '<table border="2"><tr>';

            // 테이블 헤더 생성
            var header = ['TPort Num', 'OIP Num', 'Oport Num', 'T Avg Pkt', 'O Avg Pkt', 'T Avg Bytes', 'O Avg Bytes', 'Avg_Dur', 'S Std Pkt', 'O Std Pkt', 'S Std Bytes', 'O Std Bytes', 'std_dur']
            for (var key in header) {
                tableHtml += '<th>' + header[key] + '</th>';
            }
            tableHtml += '</tr>';
            
            var data = profileData[idx];
            let count = 0;
            
            tableHtml += '<tr>';
            
            for (var key in data) {
                tableHtml += '<td>' + data[key] + '</td>';
            }
            tableHtml += '</tr>';
            tableHtml += '</table>';
            
            var flow_header = ['SIP', 'DIP', 'proto', 'Sport', 'Dport', 'Inpkt', 'Outpkt', 'Inbytes', 'Outbytes', 'Duration', 'Label']
            tableHtml += '<table border="2"><tr>';
            
            for (var key in flow_header) {
                tableHtml += '<th>' + flow_header[key] + '</th>';
            }
            tableHtml += '</tr>';
            tableHtml += '<tr>';
            
            for (var key in flowData) {
                tableHtml += '<td>' + flowData[key] + '</td>';
                count += 1;

                // 11개의 열마다 새로운 행을 시작
                if (count % 11 === 0) {
                    tableHtml += '</tr><tr>';
                }
            }
            
            tableHtml += '</tr>';
            tableHtml += '</table>';
            

            modalContent.innerHTML = tableHtml;
            modaltitleContent.innerHTML = '<h3>' + title + ' (' + totalLen + ')' + '</h3>';
            modal.style.display = 'block';

            // 모달 위치 설정
            var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            var windowWidth = window.innerWidth;
            var modalWidth = modal.offsetWidth;

            modal.style.top = (event.clientY + scrollTop) + 'px';

            if (type === 'test') {
                // 테스트 모달의 경우 오른쪽에 위치
                modal.style.left = (windowWidth - modalWidth) + 'px';
            } else {
                // 트레인 모달의 경우 기존 위치 설정 사용
                modal.style.left = '0px';
            }
        }
        function closeModal(type) {
            var modalId = type === 'train' ? 'modal-train' : 'modal-test';
            var modal = document.getElementById(modalId);
            modal.style.display = 'none';
        }
        </script>
        <style>
        .row {
            display: flex;
        }
        .column {
            flex: 50%;
            padding: 20px;
            box-sizing: border-box;
        }
        .column.test {
            text-align: right;  /* Test 컬럼의 내용을 오른쪽 정렬 */
            padding-right:20px;
        }
        .modal {
            display: none;
            position: fixed; /* fixed로 변경 */
            z-index: 1;
            left: 25%; /* 화면의 25% 지점에 위치 */
            top: 25%; /* 화면의 상단에서 25% 지점에 위치 */
            width: 50%; /* 너비를 화면의 50%로 설정 */
            height: 600px; /* 높이를 자동으로 설정 */
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        .button-benign {
            background-color: blue;
            color: white;
        }
        .button-attack {
            background-color: red;
            color: white;
        }
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 100%; /* 내용을 모달의 너비에 맞춤 */
        }
        table {
            border-collapse: collapse;
            width: 100%; /* 테이블을 모달의 너비에 맞춤 */
        }

        th, td {
            border: 1px solid black;
            text-align: left;
            padding: 8px;
        }
        td {
            background-color : white;
        }

        th {
            background-color: #f2f2f2;
        }
        
        </style>
        </head><body>'''
        for quan in self.intersection:
            # Train과 Test의 라벨 세트를 확인
            train_labels = self.train_ip.get_quan_label(quan)
            test_labels = self.test_ip.get_quan_label(quan)
            
            # 라벨에 따른 클래스 결정
            test_class = "button-attack" if "Attack" in test_labels else "button-benign"
            html_content += '<div class="row">'
            # Train Column
            html_content += '<div class="column">'
            html_content += f'<button onclick="toggleData(\'{quan}\', \'train\')">{quan} (Train)</button><br>'
            html_content += f'<div id="data-train-{quan}" style="display:none;">'
            for idx, i in enumerate(self.train_ip.get_quan_index(quan)):
                train_class = "button-attack" if train_labels[idx].upper() != 'BENIGN' else "button-benign"
                html_content += f'<button class="{train_class}" onclick="showModal({i}, \'train\', event)">{i}</button>'
            html_content += '</div>'
            html_content += '</div>'

            # Test Column
            html_content += '<div class="column test">'
            html_content += f'<button onclick="toggleData(\'{quan}\', \'test\')">{quan} (Test)</button><br>'
            html_content += f'<div id="data-test-{quan}" style="display:none;">'
            for idx, i in enumerate(self.test_ip.get_quan_index(quan)):
                test_class = "button-attack" if test_labels[idx].upper() != 'BENIGN' else "button-benign"
                html_content += f'<button class="{test_class}" onclick="showModal({i}, \'test\', event)">{i}</button>'
            html_content += '</div>'
            html_content += '</div>'
            html_content += '</div>'  # End of Row

        # Train 모달
        html_content += '''
        <div id="modal-train" class="modal">
            <div class="modal-content">
                <span onclick="closeModal('train')" class="close">&times;</span>
                <h3 id="modal-title-train"></h3>
                <p id="modal-content-train"></p>
            </div>
        </div>'''

        # Test 모달
        html_content += '''
        <div id="modal-test" class="modal">
            <div class="modal-content">
                <span onclick="closeModal('test')" class="close">&times;</span>
                <h3 id="modal-title-test"></h3>
                <p id="modal-content-test"></p>
            </div>
        </div>
        </body></html>'''
        return html_content

    def save_html(self, filename):
        html_content = self.generate_html()
        with open(filename, 'w') as file:
            file.write(html_content)

def all_do(dataset_path, min_data, attack, change_feature, add_src, separate_attackIP, count_prot, train_window, test_window, logN, real_time, using_minmax, using_quan, p0):
    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})"
    
    if using_quan == 'log':
        parameter += f'_pro({count_prot})_as({add_src})_log({logN})'
    elif using_quan == 'bayesian':
        parameter += f'_pro({count_prot})_as({add_src})_p0({p0})'
    
    global train_dataset
    global test_dataset
    global train_multi
    global test_multi
    global train_data
    global test_data
    global train_raw
    global test_raw
    global train_key
    global test_key
    
    with open(f"./debug_data/{dataset_path}/{parameter}/train_multi_dict_attack{attack}.pkl", 'rb') as f:
        train_multi = pickle.load(f)
    with open(f"./debug_data/{dataset_path}/{parameter}/test_multi_dict_attack{attack}.pkl", 'rb') as f:
        test_multi = pickle.load(f)
        
    with open(f"./debug_data/{dataset_path}/{parameter}/train_data_attack{attack}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(f"./debug_data/{dataset_path}/{parameter}/test_data_attack{attack}.pkl", 'rb') as f:
        test_data = pickle.load(f)
        
    parameter = f"cf({change_feature})_sepIP({separate_attackIP})_min({min_data})_mm({using_minmax})"

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
            
    train_dataset = dict()
    test_dataset = dict()

    train_list = os.listdir(f'./dataset/{dataset_path}/train/')
    test_list = os.listdir(f'./dataset/{dataset_path}/test/')

    for file in train_list:
        file_name = file.split('.')[0]
        data = pd.read_csv(f"./dataset/{dataset_path}/train/{file}")
        train_dataset[file_name] = data
    
    for file in test_list:
        file_name = file.split('.')[0]
        data = pd.read_csv(f"./dataset/{dataset_path}/test/{file}")
        test_dataset[file_name] = data
    
    if using_quan == 'log':
        file_name = f"log({logN})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax}).csv"
    elif using_quan == 'bayesian':
        file_name = f"p0({p0})_as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax}).csv"

    save_file = f"./result/{dataset_path}/{using_quan}_{file_name}.csv"
    data = pd.read_csv(fr'{save_file}', encoding='utf-8')
    
    if using_quan == 'log':
        parameter = f"log({logN})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax})"
    elif using_quan == 'bayesian':
        parameter = f"p0({p0})_as({add_src})-cf({change_feature})-prot({count_prot})-sepIP({separate_attackIP})-min({min_data})-atk({attack})-window({train_window}-{test_window})-mm({using_minmax})"
    
    # 여기서 고르는 방식을 바꿔야함.
    fp_data = data[(data['Test IP Label'] == 'BENIGN') & (data['Max Sim'] != '-')]
    print(len(fp_data))
    
    if not os.path.isdir(f'./Debug/{dataset_path}/'):
        os.mkdir(f'./Debug/{dataset_path}/')
    
    if not os.path.isdir(f'./Debug/{dataset_path}/{parameter}'):
        os.mkdir(f'./Debug/{dataset_path}/{parameter}')
        os.mkdir(f'./Debug/{dataset_path}/{parameter}/False-Positive')
        os.mkdir(f'./Debug/{dataset_path}/{parameter}/True-Positive')

    for index, row in fp_data.iterrows():
        # 'MAX' 컬럼에서 set을 추출하고, 'Test' 컬럼의 값을 가져옴
        max_set = eval(row['Max IP'])  # set 형식의 문자열을 실제 set 객체로 변환
        test_IP = row['Test IP']
        sim = row['Max Sim']
        
        for train_ip in max_set:
            b = IP_compare(f'{train_ip}', f'{test_IP}')
            b.save_html(f'./Debug/{dataset_path}/{parameter}/False-Positive/{sim}-{train_ip}-{test_IP}.html')
        
    # 여기서 고르는 방식을 바꿔야함.
    tp_data = data[(data['Test IP Label'] != 'BENIGN') & (data['Max Sim'] != '-')]

    for index, row in tp_data.iterrows():
        # 'MAX' 컬럼에서 set을 추출하고, 'Test' 컬럼의 값을 가져옴
        max_set = eval(row['Max IP'])  # set 형식의 문자열을 실제 set 객체로 변환
        test_IP = row['Test IP']
        sim = row['Max Sim']
        
        for train_ip in max_set:
            b = IP_compare(f'{train_ip}', f'{test_IP}')
            b.save_html(f'./Debug/{dataset_path}/{parameter}/True-Positive/{sim}-{train_ip}-{test_IP}.html')
            
if __name__ == "__main__":
    global min_data
    min_data = 7
    change_feature = False
    seperate_attackIP = True
    count_prot = True
    using_minmax = True
    add_src = True
    real_time = 0
    attack = 1 # 0이 정상 1이 공격 2가 혼합
    train_window = 0
    test_window = 10
    p0 = 0.05
    logN = 1.2
    
    for data in ['MTA']:
        print(f"{data} 시작!")
        for using_quan in ['bayesian']:
            all_do(data, min_data, attack, change_feature, add_src, seperate_attackIP, count_prot, train_window, test_window, logN, real_time, using_minmax, using_quan, p0)
        print(f"{data} 끝!")