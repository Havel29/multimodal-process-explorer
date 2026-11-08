import numpy as np
import pandas as pd
from datetime import datetime, timedelta
diagnosis_er = pd.read_csv('./dataset/diagnosis_er_02_new.csv')
diagnosis_hosp = pd.read_csv('./dataset/diagnosis_hosp_03_new.csv')

patient = pd.read_csv('dataset/patient_01.csv', encoding='unicode_escape')
patient = patient.fillna("N.A.")
list_act_adm = []
list_gender = []
list_age = []
list_diagnose = []
list_dept = []

list_in_icu_act = []
list_in_icu_time = []

list_out_icu_act = []
list_out_icu_time = []
list_out_icu_duration = []
list_out_icu_ntimes = []

list_discharge_act = []
list_discharge_outcome = []
list_discharge_time = []

list_diagnose_er = []
list_diagnose_hosp = []
list_vital_sign_ED = []
list_patient = []
list_time_ED = []
list_patient_icu = []

lista_diagnose_total = []
lista_diagnose_er_total = []

for index, row in patient.iterrows():
    list_time_ED.append(pd.to_datetime(row['admission_d_inpat']).strftime('%Y-%m-%d %H:%M:%S'))#- timedelta(seconds=1))

    list_act_adm.append('entered in ED')
    list_gender.append(row['sex'])
    list_age.append(row['age'])
    list_dept.append(row['department_emerg'])
    list_diagnose.append(row['diag_emerg'])
    dict_parameter = {'blood pressure max':row['bp_max_first_emerg'],
                      'blood pressure min':row['bp_min_first_emerg'],
                      'body temperature':row['temp_first_emerg'],
                      'heart rate':row['hr_first_emerg'],
                      'oxygen saturation':row['sat_02_first_emerg'],
                      'glucouse':row['glu_first_emerg'],
                      'diuresis':row['diuresis_first_emerg']
                      }
    list_vital_sign_ED.append(dict_parameter)

    if row['patient_id'] in diagnosis_er['patient_id'].values:
        dict_diagnosis_er = []
        matching_row = diagnosis_er[diagnosis_er['patient_id'] == row['patient_id']]
        for d in ['dia_ppal_desc', 'dia_02_desc', 'dia_03_desc', 'dia_04_desc', 'dia_05_desc', 'dia_06_desc', 'dia_07_desc', 'dia_08_desc', 'dia_09_desc', 'dia_10_desc', 'dia_11_desc', 'dia_12_desc']:
            if pd.notna(matching_row[d].values[0]):# != '':
                dict_diagnosis_er.append(matching_row[d].values[0])
                lista_diagnose_total.append(matching_row[d].values[0])
        list_diagnose_er.append(dict_diagnosis_er)
    else:
        list_diagnose_er.append({'dia_ppal':'N.A.'})

    if row['patient_id'] in diagnosis_hosp['patient_id'].values:
        dict_diagnosis_hosp = []
        matching_row = diagnosis_hosp[diagnosis_hosp['patient_id'] == row['patient_id']]
        for d in ['dia_ppal_desc', 'dia_02_desc', 'dia_03_desc', 'dia_04_desc', 'dia_05_desc', 'dia_06_desc', 'dia_07_desc', 'dia_08_desc', 'dia_09_desc', 'dia_10_desc', 'dia_11_desc', 'dia_12_desc', 'dia_13_desc', 'dia_14_desc', 'dia_15_desc', 'dia_16_desc', 'dia_17_desc', 'dia_18_desc', 'dia_19_desc']:
            if pd.notna(matching_row[d].values[0]):# != '':
                dict_diagnosis_hosp.append(matching_row[d].values[0])
                lista_diagnose_er_total.append(matching_row[d].values[0])
        list_diagnose_hosp.append(dict_diagnosis_hosp)
    else:
        list_diagnose_hosp.append(['NA'])



    if row['icu_date_in'] != 'N.A.':
        list_in_icu_act.append('entered in ICU')
        list_in_icu_time.append(pd.to_datetime(row['icu_date_in']).strftime('%Y-%m-%d %H:%M:%S'))

        list_out_icu_act.append('discharged from ICU')
        list_out_icu_time.append(pd.to_datetime(row['icu_date_out']).strftime('%Y-%m-%d %H:%M:%S'))
        list_patient_icu.append(row['patient_id'])
        list_out_icu_duration.append(row['icu_days'])
        list_out_icu_ntimes.append(row['icu_n_ing'])

    list_discharge_act.append('discharged')
    list_discharge_time.append(pd.to_datetime(row['discharge_date']).strftime('%Y-%m-%d %H:%M:%S'))# + timedelta(days=1))



    list_discharge_outcome.append(row['destin_discharge'])

    list_patient.append(row['patient_id'])

log_discharge = pd.DataFrame({"patient_id":list_patient, 'activity':list_discharge_act, "time:timestamp":list_discharge_time, "outcome":list_discharge_outcome, 'list_diagn_hosp':list_diagnose_hosp}).to_csv('./preprocessing/discharge.csv', index=False)
log_ED = pd.DataFrame({"patient_id":list_patient, 'activity':list_act_adm, "time:timestamp":list_time_ED, "gender":list_gender, "age":list_age, 'diag_emerg':list_diagnose, 'list_diagn_er':list_diagnose_er,'department_emerg':list_dept, "vital_sign_ED":list_vital_sign_ED}).to_csv('./preprocessing/enter_ED.csv', index=False)#, "BP min":list_vital_sign_ED_bp_min, "Body temperature":list_vital_sign_ED_bt, "Hearth rate": list_vital_sign_ED_hr, 'Oxygen Saturation': list_vital_sign_ED_spo2, 'Glucouse lvl': list_vital_sign_ED_gluc, 'Urine output':list_vital_sign_ED_urine})
log_ICU_in = pd.DataFrame({"patient_id":list_patient_icu, 'activity':list_in_icu_act, "time:timestamp":list_in_icu_time}).to_csv('./preprocessing/ICU_in.csv', index=False)
log_ICU_out = pd.DataFrame({"patient_id":list_patient_icu, 'activity':list_out_icu_act, "time:timestamp":list_out_icu_time, "icu_days": list_out_icu_duration, "icu_n_ing":list_out_icu_ntimes}).to_csv('./preprocessing/ICU_out.csv', index=False)
print('patient preprocessing complete')