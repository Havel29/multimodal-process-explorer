import pandas as pd
from datetime import datetime, timedelta

df_enter = pd.read_csv('./preprocessing/enter_ED.csv')
df_discharge = pd.read_csv('./preprocessing/discharge.csv')
df_labtest = pd.read_csv('./preprocessing/lab_test.csv')
df_medication = pd.read_csv('./preprocessing/medication.csv')
df_vital_signs = pd.read_csv('./preprocessing/vital_signs.csv')
df_icu_in = pd.read_csv('./preprocessing/ICU_in.csv')
df_icu_out = pd.read_csv('./preprocessing/ICU_out.csv')
df_image = pd.read_csv('./preprocessing/image.csv')


df_enter['time:timestamp'] = pd.to_datetime(df_enter['time:timestamp'])
df_discharge['time:timestamp'] = pd.to_datetime(df_discharge['time:timestamp'])
df_labtest['time:timestamp'] = pd.to_datetime(df_labtest['time:timestamp'])
df_medication['time:timestamp'] = pd.to_datetime(df_medication['time:timestamp'])
df_vital_signs['time:timestamp'] = pd.to_datetime(df_vital_signs['time:timestamp'])
df_icu_in['time:timestamp'] = pd.to_datetime(df_icu_in['time:timestamp'])
df_icu_out['time:timestamp'] = pd.to_datetime(df_icu_out['time:timestamp'])
df_image['time:timestamp'] = pd.to_datetime(df_image['time:timestamp'])

covid_log = pd.concat([df_enter, df_discharge, df_labtest, df_medication, df_vital_signs, df_icu_in, df_icu_out, df_image])
covid_log['time:timestamp'] = pd.to_datetime(covid_log['time:timestamp'])

covid_log.sort_values(by=['patient_id','time:timestamp'], ascending=True)#.to_csv('preprocessing/covid_log.csv', index=False)

#284134

fix_time = covid_log.groupby('patient_id')
list_new = []
new_log = pd.DataFrame()
for index, row in fix_time:
    sort_row = row.sort_values(by=['time:timestamp'])
    first_row = sort_row.iloc[0]
    last_row = sort_row.iloc[-1]

    sort_row.loc[sort_row['activity'] == 'entered in ED', 'time:timestamp'] = first_row['time:timestamp'] - timedelta(seconds=1)
    sort_row.loc[sort_row['activity'] == 'discharged', 'time:timestamp'] = last_row['time:timestamp'] + timedelta(days=1)
    #print(sort_row.loc[sort_row['activity'] == 'discharge', 'outcome'].values)
    sort_row['outcome'] = sort_row.loc[sort_row['activity'] == 'discharged', 'outcome'].values[0]
    list_new.append(sort_row)
    #new_log = pd.concat([new_log, sort_row])   #print(first_row)
    #print(last_row)
    #print(row)
    #exit()
new_log = pd.concat(list_new)
new_log.sort_values(by=['patient_id','time:timestamp'], ascending=True).to_csv('./preprocessing/covid_log.csv', index=False)