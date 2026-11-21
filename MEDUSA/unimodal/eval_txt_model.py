from transformers import AutoTokenizer,AutoModelForSequenceClassification
from unimodal.history_text import TextDataset
from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification
import os
from pathlib import Path
from typing import Optional
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

model_name = sys.argv[1]
model_number = str(sys.argv[2])
num_class = 2
BATCH = 32

if model_name == 'bertm':
    weights_dir = 'prajjwal1/bert-medium'
elif model_name == 'roberta':
    weights_dir = 'FacebookAI/roberta-base'
elif model_name == 'gpt2':
    weights_dir = 'openai-community/gpt2'
elif model_name == 'cbert':
    weights_dir = 'emilyalsentzer/Bio_ClinicalBERT'
else:
    print('model not found')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('../covid_log/covid_test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('../covid_log/covid_label_train.pkl', 'rb') as f:
    label_train = pickle.load(f)

with open('../covid_log/covid_label_test.pkl', 'rb') as f:
    label_test = pickle.load(f)

label2id = {}
id2label = {}
i = 0
for l in list(np.unique(label_train)):
    label2id[l] = i
    id2label[i] = l
    i = i + 1

print(label2id)

label_test_int = []
for l in label_test:
    label_test_int.append(label2id[l])



if model_name =='gpt2':

        # Get model configuration.
        print('Loading configuraiton...')
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=weights_dir, num_labels=num_class)

        # Get model's tokenizer.
        print('Loading tokenizer...')
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=weights_dir, truncation_side='left')
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token

        # Get the actual model.
        print('Loading model...')
        test_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=weights_dir, config=model_config)

        # resize model embedding to match new tokenizer
        test_model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        test_model.config.pad_token_id = test_model.config.eos_token_id
else:
        print('weight dir', weights_dir)
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
        test_model = AutoModelForSequenceClassification.from_pretrained(weights_dir, trust_remote_code=True, num_labels=num_class)

test_dataset = TextDataset(test, label_test_int, tokenizer, 512)

test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
# Load the state dictionary into the model
test_model.load_state_dict(torch.load('unimodal/model_um/covid_'+model_name+str(model_number)+'.pth'))
test_model = test_model.to(device)

test_model.eval()
all_targets = []
all_predictions = []
pred_prob = []

with torch.no_grad():  # non aggiorna i pesi
    for batch, labels in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = test_model(input_ids, attention_mask=attention_mask).logits
        pred_prob.append(output.cpu())
        predicted = output.argmax(dim=1)
        all_targets.extend(labels.to('cpu').numpy())
        all_predictions.extend(predicted.to('cpu').numpy())

all_targets = [int(x) for x in all_targets]
all_predictions = [int(x) for x in all_predictions]
report = classification_report(all_targets, all_predictions, output_dict=False, digits=4)
print(report)
