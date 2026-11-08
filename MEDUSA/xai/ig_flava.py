import torch
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from torch.utils.data import DataLoader
import torch
import pickle
import pandas as pd
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from transformers import FlavaProcessor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import cv2
import pickle
from torchmultimodal.models.flava.model import flava_model_for_classification
import sys
import matplotlib.pyplot as plt

class CustomFlavaDataset(Dataset):
    def __init__(self, image_paths, texts, labels, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #image = Image.open(self.image_paths[idx]).convert("RGB")
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
            max_length=512,
            
        )
        
        # Remove batch dimension added by processor
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
            
        return {
            "inputs": inputs,
            "labels": torch.tensor(label, dtype=torch.long)
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

num_classes = 2
n_model = sys.argv[1]
BATCH = 1

dist = (np.unique(label_train_int, return_counts=True))
class_weight = {0: ((len(label_train_int)) / (2 * dist[1][0])), 1: ((len(label_train_int)) / (2 * dist[1][1]))}
class_weights = torch.tensor([(((len(label_train_int)) / (2 * dist[1][0]))), (((len(label_train_int)) / (2 * dist[1][1])))],dtype=torch.float).to(device)
print(class_weights)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
test_model = flava_model_for_classification(num_classes=num_classes, pretrained=True, loss_fn=criterion)
test_model.to(device)

# Load the state dictionary into the model
test_model.load_state_dict(torch.load('multimodal/model_mm/covid_flava'+str(n_model)+'.pth')) #bertmedium_10.pth'))
test_model = test_model.to(device)
processor = FlavaProcessor.from_pretrained("facebook/flava-full", truncation_side='left')

test_dataset = CustomFlavaDataset(test_image, test_text, label_test_int, processor)

test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
test_model.eval()
print(test_model)

def forward_func(input_ids, attention_mask, pixel_values, labels):
    """
    Forward function for Captum that matches FLAVA model expected parameters
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)
    
    # Use named arguments that match FLAVA model signature  
    output = test_model(text=input_ids, image=pixel_values, labels=labels)
    return output.logits

def image_forward_func(pixel_values, input_ids, attention_mask, labels):
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output = test_model(text=input_ids, image=pixel_values, labels=labels)
            return output.logits


class MultimodalExplainer:
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        
    def explain_text(self, inputs, labels, n_steps, cont):
        vis_data_records = []
        #output = test_model(input_ids, attention_mask, image)
        #print(inputs)
        #exit()
        test_model.eval()
        output = test_model(text = inputs["input_ids"], image = inputs["pixel_values"], labels = labels)
        pred = output.logits.argmax(dim=1)

        
        token_reference = TokenReferenceBase(reference_token_idx=processor.tokenizer.pad_token_id)
        reference_indices = token_reference.generate_reference(512, device=device).unsqueeze(0)
        lig = LayerIntegratedGradients(forward_func, test_model.model.text_encoder.embeddings)
        #lig = LayerIntegratedGradients(test_model, test_model.model.text_encoder.embeddings)
        attributions_ig, delta = lig.attribute(
            inputs["input_ids"],                  # actual input tensor
            reference_indices,                    # baseline
            additional_forward_args=(
                inputs["attention_mask"],
                inputs["pixel_values"],
                labels
            ),
            n_steps=n_steps,
            return_convergence_delta=True,
            target=labels.item()
        )

        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        a = inputs["input_ids"].cpu().numpy().tolist()

        new_a = [processor.tokenizer.convert_ids_to_tokens(t) for t in a[0]]

        text = []
        for t in new_a:
            if t != '[CLS]' and t != '[SEP]':
                text.append(t.replace('##',''))
            else:
                text.append('')
        
        if id2label[pred.item()] == id2label[labels.item()]:
            pred_res = 'correct'
        else:
            pred_res = 'wrong'

        vis_data_records.append(visualization.VisualizationDataRecord(
            attributions,
            pred.item(),
            id2label[pred.item()],  # Label.vocab.itos[pred_ind],
            id2label[labels.item()],
            pred_res,
            attributions.sum(),
            text,
            delta))
        
        html = visualization.visualize_text(vis_data_records)
        soup = BeautifulSoup(html.data, 'html.parser')

        with open(f'XAI/txt/exp_{cont}_label_{labels.item()}.html', 'w') as f:
            f.write(str(soup))

        return attributions, pred_res



    def explain_image(self, inputs, labels, n_steps, cont):
        """Generate explanations for the visual component"""
        
        # Use consistent model calling
        output = test_model(text=inputs["input_ids"], image=inputs["pixel_values"], labels=labels)
        pred = output.logits.argmax(dim=1).item()
        
        test_model.eval()
        
        # Create baseline (zero image)
        baseline_image = torch.zeros_like(inputs["pixel_values"]).to(self.device)
        
        # FIXED: Create forward function for image attribution
        def image_forward_func(pixel_values, input_ids, attention_mask, labels):
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output = test_model(text=input_ids, image=pixel_values, labels=labels)
            return output.logits
        
        # Use image encoder layer for attribution
        lig = LayerIntegratedGradients(image_forward_func, test_model.model.image_encoder.embeddings)
        
        attributions_ig = lig.attribute(
            inputs["pixel_values"],
            baseline_image,
            additional_forward_args=(
                inputs["input_ids"],
                inputs["attention_mask"], 
                labels
            ),
            n_steps=n_steps,
            target=labels.item()
        )
        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        return attributions
    
    def visualize_image_attributions(self, image, attributions, cont):
        token_attributions = list(attributions)
        patch_attributions = np.array(token_attributions[1:])  # shape: (196,)
        print(patch_attributions.shape)
        patch_map = patch_attributions.reshape(14, 14)
        plt.imshow(patch_map, cmap='hot')
        plt.title("FLAVA Patch Attributions")
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f'XAI/img/patch_flava_{cont}.png')
        plt.show()
        plt.clf()

        heatmap_resized = cv2.resize(patch_map, (224, 224))  # From 14x14 → 224x224
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        plt.imshow(image_np)
        plt.imshow(heatmap_resized, cmap='hot', alpha=0.5)  # Overlay heatmap
        plt.title("Overlay: Image + Patch Attributions")
        plt.axis('off')
        plt.savefig(f'XAI/img/patch_flava_overlay_{cont}.png')
        plt.show()
        plt.clf()
        
explainer = MultimodalExplainer(test_model, processor, device)


cont = 0
first_element_list = []
list_text_att = []
list_img_att = []
list_labels = []
for batch in test_loader:
    inputs = batch["inputs"]
    labels = batch["labels"].to(device)
    
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    text_attributions, pred_res = explainer.explain_text(
        inputs, labels, 20, cont
        )
    
    if pred_res == 'correct':
        list_text_att.append(text_attributions)
        print("Generating image explanations...")
        image_attributions = explainer.explain_image(
                inputs, labels, n_steps=20, cont=cont
            )
        #print(len(image_attributions[1:]))
        list_labels.append(labels.item())
        
        explainer.visualize_image_attributions(inputs["pixel_values"], image_attributions, cont)
        list_img_att.append(image_attributions[1:])
    cont = cont + 1


with open('XAI/global/text_att.pkl','wb') as handle:
     pickle.dump(list_text_att, handle)
     
with open('XAI/global/img_att.pkl','wb') as handle:
     pickle.dump(list_img_att, handle)

with open('XAI/global/labels.pkl','wb') as handle:
     pickle.dump(list_labels, handle)

print('end explanation')