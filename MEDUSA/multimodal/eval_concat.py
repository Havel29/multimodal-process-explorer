from transformers import AutoModel, AutoTokenizer
from multimodal.neural_network import ConcatFusion
from torch.utils.data import DataLoader
from multimodal.history_mm import MMDataset
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os
from pathlib import Path
from transformers import AutoImageProcessor
import sys

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

def get_weight_dir(
        model_ref: str,
        *,
        model_dir=HF_DEFAULT_HOME,
        revision: str = "main", ) -> Path:
    """
    Parse model name to locally stored weights.
    Args:
        model_ref (str) : Model reference containing org_name/model_name such as 'meta-llama/Llama-2-7b-chat-hf'.
        revision (str): Model revision branch. Defaults to 'main'.
        model_dir (str | os.PathLike[Any]): Path to directory where models are stored. Defaults to value of $HF_HOME (or present directory)

    Returns:
        str: path to model weights within model directory
    """
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./covid_log/covid_train.pkl', 'rb') as f:
            train_text = pickle.load(f)

with open('./covid_log/covid_test.pkl', 'rb') as f:
            test_text = pickle.load(f)

with open('./covid_log/covid_train_img.pkl', 'rb') as f:
            train_image = pickle.load(f)

with open('./covid_log/covid_test_img.pkl', 'rb') as f:
            test_image = pickle.load(f)

with open('./covid_log/covid_label_train.pkl', 'rb') as f:
            label_train = pickle.load(f)

with open('./covid_log/covid_label_test.pkl', 'rb') as f:
            label_test = pickle.load(f)

label2id = {}
id2label = {}
i = 0
for l in list(np.unique(label_train)):
    label2id[l] = i
    id2label[i] = l
    i = i + 1

label_test_int = []
for l in label_test:
    label_test_int.append(label2id[l])

num_classes = 2
BATCH = 32
model_number = sys.argv[1]

text_model_name = 'roberta'
model_name_vis = 'vitbase'


weights_dir_txt = 'FacebookAI/roberta-base'
weights_dir_vis = 'google/vit-base-patch16-224'

tokenizer = AutoTokenizer.from_pretrained(weights_dir_txt, truncation_side='left')
model_text = AutoModel.from_pretrained(weights_dir_txt)
model_text.to(device)

preprocessor = AutoImageProcessor.from_pretrained(weights_dir_vis)
model_image = AutoModel.from_pretrained(weights_dir_vis)

test_model = ConcatFusion(model_text, model_image, num_classes).to(device)

# Load the state dictionary into the model
test_model.load_state_dict(torch.load(f'multimodal/model_mm/covid_concat_vit_roberta{model_number}.pth')) #bertmedium_10.pth'))
test_model = test_model.to(device)
test_dataset = MMDataset(test_text, test_image, label_test_int, tokenizer, 512, preprocessor)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True)
test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():  # non aggiorna i pesi
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_ids_dx = batch['image'].to(device)
        output = test_model(input_ids, attention_mask, image_ids_dx)
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(batch['label'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())
all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]

model_name = 'covid_concat_vit_roberta'
report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)
