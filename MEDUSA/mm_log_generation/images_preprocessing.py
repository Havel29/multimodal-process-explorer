import pandas as pd

image = pd.read_csv('./dataset/CDSL-1.0.0-dicom-metadata.csv')
image['time:timestamp'] = image['StudyDate'] + ' ' + image['StudyTime']
image['path'] = image['patient_group_folder_id'] +'/'+ image['patient_folder_id'] +'/'+ image['study_id'] +'/'+ image['image_id'] + '.jpg'

grouped = image.groupby('patient_id')


list_patient = []
list_time = []
list_activity = []
list_body = []
list_n_scan = []
list_path = []
for group_name, group_data in grouped:
    sort_data = group_data.sort_values(['time:timestamp', 'image_id'])
    grouped_data = sort_data.groupby('time:timestamp')
    for group_name_time, group_data_time in grouped_data:
        list_patient.append(group_name)
        list_activity.append(group_data_time['Modality'].values.tolist()[0])
        list_body.append(group_data_time['BodyPart'].values.tolist()[0])
        list_n_scan.append(len(group_data_time['path'].values.tolist()))


        list_time.append(group_data_time['time:timestamp'].values.tolist()[0])
        list_path.append(group_data_time['path'].values.tolist())

log_temp = (pd.DataFrame({"patient_id":list_patient, 'activity':list_activity, 'bodypart':list_body, 'n_scan':list_n_scan, "time:timestamp":list_time, 'path': list_path}))
log_temp['activity'].replace('CT', 'performed the computed tomography', inplace=True)
log_temp['activity'].replace('CR', 'performed the computed radiography', inplace=True)
log_temp['activity'].replace('DX', 'performed the digital radiography', inplace=True)

log_temp.to_csv('./preprocessing/image.csv', index=False)
print('images preprocessing complete')