import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from datetime import datetime, timedelta
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import os
from pathlib import Path
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

HF_DEFAULT_HOME = os.environ.get("HF_HOME", "~/.cache/huggingface/hub")

def get_weight_dir(
        model_ref: str,
        *,
        model_dir=HF_DEFAULT_HOME,
        revision: str = "main") -> Path:
    model_dir = Path(model_dir)
    assert model_dir.is_dir()
    model_path = model_dir / "--".join(["models", *model_ref.split("/")])
    assert model_path.is_dir()
    snapshot_hash = (model_path / "refs" / revision).read_text()
    weight_dir = model_path / "snapshots" / snapshot_hash
    assert weight_dir.is_dir()
    return weight_dir


class CustomMultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, processor, model_type='clip'):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.processor = processor
        self.model_type = model_type

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
            max_length=77,
        )

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return {
            "inputs": inputs,
            "labels": torch.tensor(label, dtype=torch.long)
        }


class CLIPClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, loss_fn=None):
        super(CLIPClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Linear(512, num_classes)
        self.loss_fn = loss_fn
        
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # Combine image and text embeddings
        pooled_output = outputs.image_embeds + outputs.text_embeds
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return type('obj', (object,), {'loss': loss, 'logits': logits})()


def format_time(seconds):
    """Converte secondi in formato leggibile"""
    return str(timedelta(seconds=int(seconds)))

def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name, accelerator, device):
    patience_counter = 0
    
    timing_info = {
        'epochs': [],
        'total_time': 0,
        'best_epoch': 0,
        'best_loss': float('inf')
    }
    
    training_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # TRAINING
        model.train()
        train_loss = 0.0
        train_batches = 0
        train_start = time.time()
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = batch["inputs"]
            labels = batch["labels"].to(device)
        
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"], labels=labels)
            loss = out.loss
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_time = time.time() - train_start
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # VALIDATION
        model.eval()
        val_start = time.time()
        
        with torch.no_grad():
            cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

            for batch in val_dataloader:
                inputs = batch["inputs"]
                labels = batch["labels"].to(device)
        
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                
                out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"], labels=labels)
                loss = out.loss
                cumulated_loss = cumulated_loss + loss

        val_time = time.time() - val_start
        cumulated_loss = accelerator.gather(cumulated_loss).cpu().mean().item()

        if accelerator.is_main_process:
            avg_cumulated_loss = cumulated_loss / len(val_dataloader)
            epoch_time = time.time() - epoch_start
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*80}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Time: {format_time(train_time)}")
            print(f"  Val Loss:   {avg_cumulated_loss:.4f} | Val Time:   {format_time(val_time)}")
            print(f"  Epoch Time: {format_time(epoch_time)}")
            
            timing_info['epochs'].append({
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'val_loss': float(avg_cumulated_loss),
                'train_time': train_time,
                'val_time': val_time,
                'epoch_time': epoch_time
            })
            
            if avg_cumulated_loss < best_loss:
                accelerator.set_trigger()
                best_loss = avg_cumulated_loss
                patience_counter = 0
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), f'multimodal/model_mm/{model_name}_{epoch + 1}.pth')
                print(f"  ✓ Model saved (Best loss: {best_loss:.4f})")
                timing_info['best_epoch'] = epoch + 1
                timing_info['best_loss'] = float(best_loss)
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            scheduler.step(avg_cumulated_loss)
        
        if patience_counter >= patience:
            print(f"\n{'='*80}")
            print(f"Early stopping at epoch {epoch + 1}")
            print(f"{'='*80}")
            break

        accelerator.wait_for_everyone()
    
    total_time = time.time() - training_start
    timing_info['total_time'] = total_time
    
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total Training Time: {format_time(total_time)}")
        print(f"Best Epoch: {timing_info['best_epoch']}")
        print(f"Best Validation Loss: {timing_info['best_loss']:.4f}")
        print(f"Average Time per Epoch: {format_time(total_time / len(timing_info['epochs']))}")
        print(f"{'='*80}\n")
        
        with open(f'multimodal/model_mm/{model_name}_timing.json', 'w') as f:
            json.dump(timing_info, f, indent=2)
        print(f"✓ Timing info saved: multimodal/model_mm/{model_name}_timing.json\n")


if __name__ == '__main__':
    seed = 42
    LEARNING_RATE = 1e-5
    BATCH = 16
    MODEL_TYPE = 'clip' 
    set_seed(seed)
    
    print("Loading data...")
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

    print(f"Train size: {len(train_text)}, Test size: {len(test_text)}")

    label2id = {}
    id2label = {}
    for i, l in enumerate(np.unique(label_train)):
        label2id[l] = i
        id2label[i] = l

    label_train_int = [label2id[l] for l in label_train]
    label_test_int = [label2id[l] for l in label_test]

    X_train_text, X_val_text, X_train_image, X_val_image, y_train, y_val = train_test_split(
        train_text, train_image, label_train_int,
        test_size=0.2, random_state=42, shuffle=True,
        stratify=label_train_int)

    print('TRAINING START...')
    from accelerate import Accelerator

    accelerator = Accelerator(split_batches=False)
    device = accelerator.device

    if MODEL_TYPE == 'clip':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", truncation_side='left')
        model_name = 'covid_clip'

    train = CustomMultimodalDataset(X_train_image, X_train_text, y_train, processor, MODEL_TYPE)
    validation = CustomMultimodalDataset(X_val_image, X_val_text, y_val, processor, MODEL_TYPE)

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)    
    val_loader = DataLoader(validation, batch_size=BATCH, shuffle=False)

    dist = np.unique(label_train_int, return_counts=True)
    class_weights = torch.tensor(
        [len(label_train_int) / (2 * count) for count in dist[1]],
        dtype=torch.float
    ).to(device)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_labels = len(np.unique(label_train_int))
    if MODEL_TYPE == 'clip':
        model = CLIPClassifier(num_labels, pretrained=True, loss_fn=criterion)
    
    model.to(device)
    print(f"Device: {device}")
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler)
    
    pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, 3, 15, float('inf'), model_name, accelerator, device)
