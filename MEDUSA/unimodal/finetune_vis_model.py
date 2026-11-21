import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModelForImageClassification
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time            
from transformers import AutoModelForImageClassification
from unimodal.history_image import CustomImageDataset
from transformers import AutoImageProcessor
from accelerate import Accelerator
import os
from pathlib import Path
import sys
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)

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

def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)  # Assuming labels are 0 or 1
                output = model(inputs).logits
                loss = criterion(output, labels)
                accelerator.backward(loss)
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for inputs, labels in val_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)  # Assuming labels are 0 or 1
                    output = model(inputs).logits
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

    set_seed(seed)

    with open('../covid_log/covid_train_img.pkl', 'rb') as f:
        train = pickle.load(f)

    with open('../covid_log/covid_test_img.pkl', 'rb') as f:
        test = pickle.load(f)

    with open('../covid_log/covid_label_train.pkl', 'rb') as f:
        label_train = pickle.load(f)

    with open('../covid_log/covid_label_test.pkl', 'rb') as f:
        label_test = pickle.load(f)
    print(np.unique(label_train, return_counts=True))
    print(np.unique(label_test, return_counts=True))
    
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
                                                    train, label_train_int,
                                                      test_size=0.2, random_state=42, shuffle=True,
                                                      stratify=label_train_int)

    if model_name == 'vit':
        weights_dir = 'google/vit-base-patch16-224'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )
    
    elif model_name == 'deit':
        weights_dir = 'facebook/deit-small-patch16-224'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )

    elif model_name == 'swin':
         weights_dir= 'microsoft/swin-base-patch4-window7-224-in22k'
         preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
         model = AutoModelForImageClassification.from_pretrained(
            weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True
        )
        
    elif model_name == 'resnet':
        weights_dir = 'microsoft/resnet-101'
        preprocessor = AutoImageProcessor.from_pretrained(weights_dir)
        model = AutoModelForImageClassification.from_pretrained(weights_dir,
            num_labels=num_class,
            ignore_mismatched_sizes=True)
    else:
        print('model not found')

    print('TRAINING START...')

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    model = model.to(device)
    print(model)

    dataset_train = CustomImageDataset(X_train, y_train, transform=preprocessor)
    dataset_test = CustomImageDataset(X_val, y_val, transform=preprocessor)

    train_loader = DataLoader(dataset_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(dataset_test, batch_size=BATCH, shuffle=False)

    print('device-->', device)

    dist = (np.unique(label_train_int, return_counts=True))
    class_weight = {0: ((len(label_train_int)) / (2 * dist[1][0])), 1: ((len(label_train_int)) / (2 * dist[1][1]))}
    class_weights = torch.tensor([(((len(label_train_int)) / (2 * dist[1][0]))), (((len(label_train_int)) / (2 * dist[1][1])))],dtype=torch.float).to(device)
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    startTime = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, 3, 15, float('inf'), 'covid_'+model_name)