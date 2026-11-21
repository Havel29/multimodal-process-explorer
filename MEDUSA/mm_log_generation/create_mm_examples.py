import pandas as pd
import pickle
from jinja2 import Template
from mm_log_generation.skeleton import skeleton
import numpy as np
from PIL import Image
path_folder = '/home/vincenzo/Scaricati/covid_log/IMAGES/'
def extract_timestamp_features(group):
    timestamp_col = 'time:timestamp'
    group = group.sort_values(timestamp_col, ascending=True)
    # end_date = group[timestamp_col].iloc[-1]
    start_date = group[timestamp_col].iloc[0]

    timesincelastevent = group[timestamp_col].diff()
    timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
    group["timesincelastevent"] = timesincelastevent.apply(
        lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

    elapsed = group[timestamp_col] - start_date
    elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
    group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds
    return group

def history_conversion(csv_log):
        group_csv = csv_log.groupby('patient_id', sort=False)
        list_label = []
        list_img = []
        list_seq = []
        list_act_total = []
        z = 0
        for group_name, group_data in group_csv:
            event_dict_hist = {}
            event_text = ''
            list_act_temp = []
            for index, row in group_data.iterrows():
                for v in skeleton['feature']:
                    event_dict_hist[v] = row[v]
                event_template = Template(skeleton[row['activity']])
                list_act_temp.append(row['activity'])
                if row['activity'] == 'entered in ED' or row['activity'] == 'performed the laboratory tests' or row['activity'] == 'received vital signs check'\
                        or row['activity'] == 'was dispensed medicine' or row['activity'] == 'discharged':
                    event_text = event_text + ' ' + event_template.render(event_dict_hist).replace('{','').replace('}','').replace("'",'').replace('[','').replace(']','')
                else:
                    event_text = event_text + event_template.render(event_dict_hist)

                if row['activity'] == 'performed the computed radiography' or row['activity'] == 'performed the digital radiography':
                    path_img = row['path'].split(',')[-1].replace(']', '').replace('[', '').replace(' ', '').replace("'",
                                                                                                                      '')
                    im = Image.open(path_folder + path_img)
                    new_image = im.resize((224, 224))
                    img_cr = np.array(new_image)
                    if row['outcome'] == 'Home' or row['outcome'] == 'Death':
                        list_seq.append(event_text)
                        list_img.append(img_cr)
                        list_label.append(row['outcome'])
                        list_act_total.append(list_act_temp.copy())
                z = z + 1
        return list_seq, list_img, list_label, list_act_total


covid_log = pd.read_csv('./preprocessing/covid_log.csv')
covid_log['time:timestamp'] = pd.to_datetime(covid_log['time:timestamp'])
covid_log = covid_log.groupby('patient_id', group_keys=False).apply(extract_timestamp_features)
covid_log = covid_log.reset_index(drop=True)
grouped = covid_log.groupby("patient_id")


value_to_check = 'Absconded'
value_to_check2 = 'N.A.'

list_df_filtered = []
for group_name, group_data in grouped:
    is_present = value_to_check in group_data['outcome'].values
    is_present2 = value_to_check2 in group_data['outcome'].values

    if is_present == False and is_present2 == False:
        list_df_filtered.append(group_data)

new_log = pd.concat(list_df_filtered)
covid_log = new_log.groupby('patient_id', group_keys=False).apply(extract_timestamp_features)
covid_log = covid_log.reset_index(drop=True)
grouped = covid_log.groupby("patient_id")

start_timestamps = grouped["time:timestamp"].min().reset_index()
start_timestamps = start_timestamps.sort_values("time:timestamp", ascending=True, kind="mergesort")
train_ids = list(start_timestamps["patient_id"])[:int(0.66 * len(start_timestamps))]
train = covid_log[covid_log["patient_id"].isin(train_ids)].sort_values("time:timestamp", ascending=True,
                                                                             kind='mergesort')
test = covid_log[~covid_log["patient_id"].isin(train_ids)].sort_values("time:timestamp", ascending=True,
                                                                              kind='mergesort')

list_seq_train, list_img_train, list_label_train, list_act_total_train = history_conversion(train)

with open('./covid_log/covid_train.pkl', 'wb') as f:
    pickle.dump(list_seq_train, f)

with open('./covid_log/covid_train_img.pkl', 'wb') as f:
    pickle.dump(list_img_train, f)

with open('./covid_log/covid_label_train.pkl', 'wb') as f:
    pickle.dump(list_label_train, f)

list_seq_train.clear()
list_img_train.clear()
list_label_train.clear()
print('Train examples complete')

list_seq_test, list_img_test, list_label_test, list_act_total_test = history_conversion(test)

with open('./covid_log/covid_test.pkl', 'wb') as f:
    pickle.dump(list_seq_test, f)

with open('./covid_log/covid_test_img.pkl', 'wb') as f:
    pickle.dump(list_img_test, f)

with open('./covid_log/covid_label_test.pkl', 'wb') as f:
    pickle.dump(list_label_test, f)

list_seq_test.clear()
list_img_test.clear()
list_label_test.clear()
print('Testing examples complete')
