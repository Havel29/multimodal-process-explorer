<table border="0">
  <tr>
    <td valign="middle" width="200">
<img width="1024" height="1024" alt="medusa" src="https://github.com/user-attachments/assets/ca2b5b89-3e79-4cf4-b4d9-ae811fc389cd" />
    </td>
    <td align="right" valign="middle">
      <h1>Multimodal Predictive Process Monitoring and its Application to Explainable Clinical Pathways</h1>
    </td>
  </tr>
</table>

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Ivan Donadello, Annalisa Appice, Oswald Lanz, Fabrizio Maria Maggi, Giuseppe Fiameni, Donato Malerba*


[*Multimodal Predictive Process Monitoring and its Application to Explainable Clinical Pathways*](https://www.sciencedirect.com/science/article/pii/S0306437926000128)

Please cite our work if you find it useful for your research and work.

```
@article{PASQUADIBISCEGLIE2026102698,
title = {Multimodal predictive process monitoring and its application to explainable clinical pathways},
journal = {Information Systems},
volume = {139},
pages = {102698},
year = {2026},
issn = {0306-4379},
doi = {https://doi.org/10.1016/j.is.2026.102698},
author = {Vincenzo Pasquadibisceglie and Ivan Donadello and Annalisa Appice and Oswald Lanz and Fabrizio Maria Maggi and Giuseppe Fiameni and Donato Malerba}
}
```

# How to Use

To run the scripts in this repository, you first need to download the **CDLS dataset** from the following link:  
👉 https://physionet.org/content/covid-data-shared-learning/1.0.0/  

Please make sure to follow all requirements specified by PhysioNet.

---

## Step 1: Prepare the Dataset
After downloading, copy the following files into your dataset folder:
atc5.csv, diagnosis_hosp_03.csv, patient_01.csv,
atc7.csv vital_signs_04.csv,
CDSL-1.0.0-dicom-metadata.csv, icd10_codes_dict.txt,
diagnosis_er_02.csv lab_06.csv,
medication_05.csv


## Step 2: Generate the Multimodal Event Log
```
./generation.sh
```
## Step 3: Extract Labeled Image–Text Pairs
Once the multimodal event log has been generated, specify the image folder path inside **create_mm_examples.py**.  
For example: path_folder = '/home/covid_log/IMAGES/' <br>
Then run:
```
python -m mm_log_generation.create_mm_examples
```

## Step 4: Fine-Tune Pre-Trained Models

Once the labelled examples have been generated, you can fine-tune the pre-trained models.

### Unimodal Text Model
```
python -m unimodal.finetune_txt_model roberta
```
### Unimodal Vision Model
```
python -m unimodal.finetune_vis_model vit
```
### Multimodal Model
```
python -m multimodal.finetune_concat
python -m multimodal.finetune_flava
```
## Multimodal ML Model
- Embeddings generation:
```
  python -m multimodal_ml.embeddings_generation
```
## Step 5: MM explanations
- Integrated gradient:
```
  python -m xai.ig_flava model_number
```
