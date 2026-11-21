#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/MEDUSA"
mkdir -p preprocessing

python -m mm_log_generation.diagnosis_preprocessing
python -m mm_log_generation.patient_preprocessing
python -m mm_log_generation.labtests_preprocessing
python -m mm_log_generation.medications_preprocessing
python -m mm_log_generation.vitalsigns_preprocessing
python -m mm_log_generation.images_preprocessing
python -m mm_log_generation.merge_df

echo "Event log written to $repo_root/MEDUSA/preprocessing/covid_log.csv"

