import pandas as pd

lab_test = pd.read_csv('./dataset/lab_06.csv')
grouped = lab_test.groupby('patient_id')
unique_lab_test = lab_test['item_lab'].unique()

list_patient = []
list_item_lab = []
list_lab = []
list_activity = []
list_avg_dose = []
list_time = []

list_val = []
list_text = []
list_ud = []
list_ref_val = []
for group_name, group_data in grouped:
    sort_data = group_data.sort_values('lab_date')
    grouped_data = sort_data.groupby('lab_number')

    for group_name_time, group_data_time in grouped_data:
        list_patient.append(group_name)
        list_lab.append(group_name_time)
        list_time.append(group_data_time['lab_date'].values.tolist()[0])
        list_activity.append('performed the laboratory tests')

        itemlab = group_data_time['item_lab'].values.tolist()
        val_result = group_data_time['val_result'].values.tolist()
        result_text = group_data_time['result_text'].values.tolist()
        ud_result = group_data_time['ud_result'].values.tolist()
        ref_values = group_data_time['ref_values'].values.tolist()

        result_dict = {}
        for item in group_data_time['item_lab'].values.tolist():#unique_lab_test:
            result_dict[item] = {'value': '', 'notes': '', 'unit of measurement': '', 'reference range values': ''}

        for a,b,c,d,e in zip(itemlab, val_result, result_text, ud_result, ref_values):
            result_dict[a] = {'value': b, 'notes': c, 'unit of measurement': d, 'reference range values': e}
            result_dict[a] = {k: 'NA' if pd.isna(v) else v for k, v in result_dict[a].items()}
        list_item_lab.append(result_dict)
        

log_lab = pd.DataFrame({"patient_id":list_patient, 'activity':list_activity, "time:timestamp":list_time, "lab":list_lab, #"lab item": list_item_lab,
                        #"val_result": list_val,
                        'lab_items': list_item_lab #"result_text": list_text, "ud_result": list_ref_val, "ref_values": list_ref_val
                        }).sort_values(['patient_id', 'time:timestamp']).to_csv('./preprocessing/lab_test.csv', index=False)

print('labtests preprocessing complete')