import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from PIL import Image
from transformers import FlavaProcessor
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchmultimodal.models.flava.model import flava_model_for_classification

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
        
        # Process inputs with the VILT processor
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


def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch["inputs"]
                labels = batch["labels"].to(device)
            
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                out = model(text = inputs["input_ids"], image = inputs["pixel_values"], labels = labels)
                loss = out.loss
                accelerator.backward(loss)
                optimizer.step()
            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for batch in val_dataloader:
                    inputs = batch["inputs"]
                    labels = batch["labels"].to(device)
            
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                    out = model(text = inputs["input_ids"], image = inputs["pixel_values"], labels = labels)
                    loss = out.loss
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
                    accelerator.save(unwrapped_model.state_dict(), 'model_mm/' + model_name + str(epoch + 1) + '.pth')
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

    set_seed(seed)
    with open('../covid_log/covid_train.pkl', 'rb') as f:
        train_text = pickle.load(f)

    with open('../covid_log/covid_test.pkl', 'rb') as f:
        test_text = pickle.load(f)

    with open('../covid_log/covid_train_img.pkl', 'rb') as f:
        train_image = pickle.load(f)

    with open('../covid_log/covid_test_img.pkl', 'rb') as f:
        test_image = pickle.load(f)

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


    X_train_text, X_val_text, X_train_image, X_val_image, y_train, y_val = train_test_split(
                                                    train_text, train_image, label_train_int,
                                                      test_size=0.2, random_state=42, shuffle=True,
                                                      stratify=label_train_int)

    model_name = 'flava'
    print('TRAINING START...')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from accelerate import Accelerator

    accelerator = Accelerator(split_batches=False)
    device = accelerator.device

    processor = FlavaProcessor.from_pretrained("facebook/flava-full", truncation_side='left')


    train = CustomFlavaDataset(X_train_image, X_train_text, y_train, processor)
    validation = CustomFlavaDataset(X_val_image, X_val_text, y_val, processor)

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)    
    val_loader = DataLoader(validation, batch_size=BATCH, shuffle=False)


    dist = (np.unique(label_train_int, return_counts=True))
    class_weight = {0: ((len(label_train_int)) / (2 * dist[1][0])), 1: ((len(label_train_int)) / (2 * dist[1][1]))}
    class_weights = torch.tensor([(((len(label_train_int)) / (2 * dist[1][0]))), (((len(label_train_int)) / (2 * dist[1][1])))],dtype=torch.float).to(device)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_labels = 2
    model = flava_model_for_classification(num_classes=num_labels, pretrained=True, loss_fn=criterion)
    print(model)

    model.to(device)

    print('device-->', device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    startTime = time.time()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, 3, 15, float('inf'), 'covid_'+model_name)
