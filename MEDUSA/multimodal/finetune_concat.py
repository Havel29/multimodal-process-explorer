import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from multimodal.history_mm import MMDataset
from multimodal.neural_network import ConcatFusion
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using CUDA
    np.random.seed(seed)
    random.seed(seed)

def pre_train(model, optimizer, train_dataloader, val_dataloader, scheduler, patience, epochs, best_loss, model_name):
        # Training loop
        patience_counter = 0
        for epoch in range(epochs):  # number of epochs
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image = batch['image'].to(device)
                labels = batch['label'].to(device)  # Assuming labels are 0 or 1
                output = model(input_ids, attention_mask, image)
                loss = criterion(output, labels)
                accelerator.backward(loss)
                optimizer.step()

            # Validation phase
            model.eval()
            with torch.no_grad():
                cumulated_loss = torch.as_tensor([0.0]).to(accelerator.device)

                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    image = batch['image'].to(device)
                    labels = batch['label'].to(device) 
                    output = model(input_ids,attention_mask,image)
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
                    accelerator.save(unwrapped_model.state_dict(), 'multimodal/model_mm/' + model_name + str(epoch + 1) + '.pth')
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

    model_name = 'concat_vit_roberta'
    frozen = False

    weights_dir_txt = 'FacebookAI/roberta-base'
    weights_dir_vis = 'google/vit-base-patch16-224'


    print('TRAINING START...')
    from accelerate import Accelerator

    accelerator = Accelerator(split_batches=True)
    device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(weights_dir_txt, truncation_side='left')
    model_text = AutoModel.from_pretrained(weights_dir_txt)

    preprocessor = AutoImageProcessor.from_pretrained(weights_dir_vis)
    model_image = AutoModel.from_pretrained(weights_dir_vis)
    
    if frozen == True:
        for param in model_text.parameters():
            param.requires_grad = False
        
        for param in model_image.parameters():
            param.requires_grad = False
        
    model = ConcatFusion(model_text, model_image, 2).to(device)

    train_dataset = MMDataset(X_train_text, X_train_image, y_train, tokenizer,512, preprocessor)
    val_dataset = MMDataset(X_val_text, X_val_image, y_val, tokenizer,512, preprocessor)

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