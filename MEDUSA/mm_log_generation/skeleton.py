skeleton = { 'feature':['timesincecasestart','gender','age', 'diag_emerg','list_diagn_er','vital_sign_ED','list_diagn_hosp','lab','lab_items', 'medicine', 'vital_sign', 'icu_days', 'icu_n_ing', 'bodypart', 'n_scan'],

            'entered in ED': '{{gender}} patient, {{age}} years old entered in ED. Vital signs were: {{vital_sign_ED}}. Diagnosis was {{diag_emerg}} detailed as: {{list_diagn_er}}.',
            'performed the laboratory tests': 'Patient performed laboratory tests: {{lab_items}} in laboratory {{lab}} {{timesincecasestart}} seconds ago.',
            'received vital signs check' : 'Patient received vital signs check: {{vital_sign}} {{timesincecasestart}} seconds ago.',
            'was dispensed medicine': 'Patient was dispensed medicine:  {{medicine}} {{timesincecasestart}} seconds ago.',
            'discharged': 'Patient discharged with diagnosis: {{list_diagn_hosp}} {{timesincecasestart}} seconds ago.',
            'entered in ICU':'Patient entered in ICU {{timesincecasestart}} seconds ago.',
            'discharged from ICU': 'Patient discharged from ICU {{timesincecasestart}} seconds ago after being admitted for {{icu_n_ing}} times along {{icu_days}} days.',
            'performed the computed tomography':'Patient performed the computed tomography of {{bodypart}} for {{n_scan}} scans {{timesincecasestart}} seconds ago.',
            'performed the computed radiography':'Patient performed the computed radiography of {{bodypart}} for {{n_scan}} scans {{timesincecasestart}} seconds ago.',
            'performed the digital radiography':'Patient performed the digital radiography of {{bodypart}} for {{n_scan}} scans {{timesincecasestart}} seconds ago.',
            }