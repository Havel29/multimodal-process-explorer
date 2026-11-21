from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
from transformers import FlavaProcessor
import sys
from torchmultimodal.models.flava.model import flava_model_for_classification


HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

class CustomFlavaDataset(Dataset):
    def __init__(self, image_paths, texts, labels, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb_img = np.stack([self.image_paths[idx]] * 3, axis=-1)
        image = Image.fromarray(rgb_img.astype(np.uint8))
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Remove batch dimension added by processor
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return {
            "inputs": inputs,
            "labels": torch.tensor(label, dtype=torch.long)
        }

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

label_train_int = []
for l in label_train:
        label_train_int.append(label2id[l])

label_test_int = []
for l in label_test:
        label_test_int.append(label2id[l])

BATCH = 32

processor = FlavaProcessor.from_pretrained("facebook/flava-full", truncation_side='left')


test = CustomFlavaDataset(test_image, test_text, label_test_int, processor)

test_loader = DataLoader(test, batch_size=BATCH, shuffle=False)    

dist = (np.unique(label_train_int, return_counts=True))
class_weight = {0: ((len(label_train_int)) / (2 * dist[1][0])), 1: ((len(label_train_int)) / (2 * dist[1][1]))}
class_weights = torch.tensor([(((len(label_train_int)) / (2 * dist[1][0]))), (((len(label_train_int)) / (2 * dist[1][1])))],dtype=torch.float).to(device)
print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
n_model = sys.argv[1]
num_labels = 2  # Binary classification
test_model = flava_model_for_classification(num_classes=num_labels, pretrained=True, loss_fn=criterion)
test_model.to(device)
# Load the state dictionary into the model
test_model.load_state_dict(torch.load('multimodal/model_mm/covid_flava'+str(n_model)+'.pth'))
test_model = test_model.to(device)

test_model.eval()

all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():  # non aggiorna i pesi
    for batch in test_loader:
        inputs = batch["inputs"]
        labels = batch["labels"].to(device)
            
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        output = test_model(text = inputs["input_ids"], image = inputs["pixel_values"], labels = labels)
        pred_prob.append(output.logits.cpu())
        predicted = output.logits.argmax(dim=1)
        all_targets.extend(batch['labels'].to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())
all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]

model_name = 'flava'
report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)
