from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import os
from pathlib import Path
from transformers import AutoModelForImageClassification, AutoImageProcessor
from unimodal.history_image import CustomImageDataset
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

model_name = sys.argv[1]
model_number = str(sys.argv[2])

with open('../covid_log/covid_test_img.pkl', 'rb') as f:
        test = pickle.load(f)

with open('../covid_log/covid_label_train.pkl', 'rb') as f:
        label_train = pickle.load(f)

with open('../covid_log/covid_label_test.pkl', 'rb') as f:
        label_test = pickle.load(f)

label2id = {}
id2label = {}
BATCH = 32
num_class = 2

i = 0
for l in list(np.unique(label_train)):
    label2id[l] = i
    id2label[i] = l
    i = i + 1

label_test_int = []
for l in label_test:
    label_test_int.append(label2id[l])

if model_name == 'vit':
        weights_dir = 'google/vit-base-patch16-224'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        test_model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )

elif model_name == 'resnet':
        weights_dir = 'microsoft/resnet-101'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        test_model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )
elif model_name == 'swin':
         weights_dir= 'microsoft/swin-base-patch4-window7-224-in22k'
         preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
         test_model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )
elif model_name == 'deit':
        weights_dir = 'facebook/deit-small-patch16-224'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        test_model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )
else:
    print('model not found')

dataset_test = CustomImageDataset(test, label_test_int, transform=preprocessor)
test_loader = DataLoader(dataset_test, batch_size=BATCH, shuffle=True)

# Load the state dictionary into the model
test_model.load_state_dict(torch.load('unimodal/model_um/covid_'+model_name+str(model_number)+'.pth'))
test_model = test_model.to(device)

test_model.eval()
all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():  # non aggiorna i pesi
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        output = test_model(inputs).logits
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(labels.to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

# Generate classification report
all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]
report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)
