import pandas as pd


icd_10_code = pd.read_csv('./dataset/icd10_codes_dict.txt', sep='|')
diagnosis_er = pd.read_csv('./dataset/diagnosis_er_02.csv')
diagnosis_hosp = pd.read_csv('./dataset/diagnosis_hosp_03.csv')
diagnosis_er['dia_ppal'] = diagnosis_er['dia_ppal'].str.replace('.', '', regex=False)
diagnosis_hosp['dia_ppal'] = diagnosis_hosp['dia_ppal'].str.replace('.', '', regex=False)

def find_column3_value(value, df2):
    match = df2[df2['code'] == value]
    return match['description1'].values[0] if not match.empty else None

for j in ['dia_ppal','dia_02','dia_03','dia_04','dia_05','dia_06','dia_07','dia_08','dia_09','dia_10','dia_11','dia_12']:
    diagnosis_er[j+'_desc'] = diagnosis_er[j].apply(lambda x: find_column3_value(x, icd_10_code))

for j in ['dia_ppal','dia_02','dia_03','dia_04','dia_05','dia_06','dia_07','dia_08','dia_09','dia_10','dia_11','dia_12','dia_13','dia_14','dia_15','dia_16','dia_17','dia_18','dia_19']:
    diagnosis_hosp[j + '_desc'] = diagnosis_hosp[j].apply(lambda x: find_column3_value(x, icd_10_code))


diagnosis_er.to_csv('./dataset/diagnosis_er_02_new.csv', index=False)
diagnosis_hosp.to_csv('./dataset/diagnosis_hosp_03_new.csv', index=False)
print('diagnosis preprocessing complete')
