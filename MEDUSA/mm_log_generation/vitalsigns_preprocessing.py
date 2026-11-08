import numpy as np
import pandas as pd
vital_signs = pd.read_csv('./dataset/vital_signs_04.csv', encoding='unicode_escape')
vital_signs = vital_signs.fillna("N.A.")

vital_signs['constants_ing_date'] = pd.to_datetime(vital_signs['constants_ing_date']).dt.date.astype(str)

vital_signs['time:timestamp'] = pd.to_datetime(vital_signs['constants_ing_date'] + ' ' + vital_signs['constants_ing_time'])
vital_signs['activity'] = 'received vital signs check'
#bp_max_ing,bp_min_ing,temp_ing,hr_ing,sat_02_ing,glu_ing
list_vital_sign = []
for index, row in vital_signs.iterrows():
    dict_parameter = {'blood pressure max': row['bp_max_ing'],
                      'blood pressure min': row['bp_min_ing'],
                      'body temperature': row['temp_ing'],
                      'heart rate': row['hr_ing'],
                      'oxygen saturation': row['sat_02_ing'],
                      'glucouse': row['glu_ing']
                      }
    list_vital_sign.append(dict_parameter)

vital_signs['vital_sign'] = list_vital_sign
vital_signs[['patient_id','activity', 'time:timestamp', 'vital_sign']].to_csv('./preprocessing/vital_signs.csv', index=False)
print('vitalsigns preprocessing complete')