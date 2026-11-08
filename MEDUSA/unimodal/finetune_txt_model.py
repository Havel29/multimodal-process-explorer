import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from unimodal.history_text import TextDataset
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from transformers import GPT2Config, GPT2Tokenizer, AutoModelForSequenceClassification, GPT2ForSequenceClassification
import os
from pathlib import Path
from accelerate import Accelerator
import sys

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)


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

def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)  # Assuming labels are 0 or 1
                output = model(input_ids, attention_mask=attention_mask).logits
                loss = criterion(output, labels)
                accelerator.backward(loss)
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for inputs, labels in val_dataloader:
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    labels = labels.to(device)  # Assuming labels are 0 or 1
                    output = model(input_ids, attention_mask=attention_mask).logits
                    loss = criterion(output, labels)
                    cumulated_loss = cumulated_loss + loss

            cumulated_loss = accelerator.gather(cumulated_loss).cpu().mean().item()

            if accelerator.is_main_process:
                avg_cumulated_loss = cumulated_loss / len(val_dataloader)
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_cumulated_loss:.4f}")
                if avg_cumulated_loss < best_loss:
                    accelerator.set_trigger()
                    best_loss = avg_cumulated_loss
                    patience_counter = 0
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), 'unimodal/model_um/'+model_name + str(epoch + 1) + '.pth')
                else:
                    patience_counter += 1
                scheduler.step(avg_cumulated_loss)
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            accelerator.wait_for_everyone()


if __name__ == '__main__':
    seed = 42
    LEARNING_RATE = 1e-5
    BATCH = 16
    num_class = 2
    model_name = sys.argv[1]
    print(model_name)

    set_seed(seed)
    with open('../covid_log/covid_train.pkl', 'rb') as f:
        train_text = pickle.load(f)

    print(len(train_text))

    with open('../covid_log/covid_test.pkl', 'rb') as f:
        test_text = pickle.load(f)

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

    label_train_int = []
    for l in label_train:
        label_train_int.append(label2id[l])

    label_test_int = []
    for l in label_test:
        label_test_int.append(label2id[l])

    X_train, X_val, y_train, y_val = train_test_split(
                                                    train_text, label_train_int,
                                                      test_size=0.2, random_state=42, shuffle=True,
                                                      stratify=label_train_int)

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

    print('TRAINING START...')

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

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
        model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=weights_dir, config=model_config)

        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
    else:
        print('weight dir', weights_dir)
        tokenizer = AutoTokenizer.from_pretrained(weights_dir, truncation_side='left')
        model = AutoModelForSequenceClassification.from_pretrained(weights_dir, trust_remote_code=True, num_labels=num_class)
        model = model.to(device)
   

    train_dataset = TextDataset(X_train, y_train, tokenizer, 512)
    val_dataset = TextDataset(X_val, y_val, tokenizer, 512)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    print('device-->', device)
    print(model)
    dist = (np.unique(label_train_int, return_counts=True))
    class_weight = {0: ((len(label_train_int)) / (2 * dist[1][0])), 1: ((len(label_train_int)) / (2 * dist[1][1]))}
    class_weights = torch.tensor([(((len(label_train_int)) / (2 * dist[1][0]))), (((len(label_train_int)) / (2 * dist[1][1])))],dtype=torch.float).to(device)
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    startTime = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, 3, 15, float('inf'), 'covid_'+model_name)