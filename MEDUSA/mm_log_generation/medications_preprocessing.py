import pandas as pd

medication = pd.read_csv('./dataset/medication_05.csv', encoding='unicode_escape')
medication['duration'] = (pd.to_datetime(medication['drug_end_date']) - pd.to_datetime(medication['drug_start_date'])).dt.days
grouped = medication.groupby('patient_id')
unique_medication = medication['drug_comercial_name'].unique()

list_med = []
list_patient = []
list_duration = []
list_time = []
list_activity = []
list_avg_dose = []
for group_name, group_data in grouped:
    sort_data = group_data.sort_values(['drug_start_date', 'medication_id'])
    grouped_data = sort_data.groupby('drug_start_date')
    for group_name_time, group_data_time in grouped_data:
        #list_med.append(group_data_time['drug_comercial_name'].values.tolist())
        list_patient.append(group_name)
        list_time.append(group_name_time)
        #list_duration.append(group_data_time['duration'].values.tolist())
        list_activity.append('was dispensed medicine')
        #list_avg_dose.append(group_data_time['daily_avrg_dose'].values.tolist())
        result_dict = {}
        for item in group_data_time['drug_comercial_name'].values.tolist():#unique_medication:
            result_dict[item] = {'average daily dose':'', 'duration': ''}
        med = group_data_time['drug_comercial_name'].values.tolist()
        daily_avrg_dose = group_data_time['daily_avrg_dose'].values.tolist()
        duration = group_data_time['duration'].values.tolist()

        for a, b, c in zip(med, daily_avrg_dose, duration):
            # print(a, b, c, d, e)
            result_dict[a] = {'average daily dose': b, 'duration': c}
        list_med.append(result_dict)

log_temp = pd.DataFrame({"patient_id":list_patient, 'activity':list_activity, "time:timestamp":list_time, "medicine":list_med}).to_csv('./preprocessing/medication.csv', index=False)
print('medications preprocessing complete')