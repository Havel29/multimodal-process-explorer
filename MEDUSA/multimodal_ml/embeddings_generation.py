from transformers import AutoModel,AutoTokenizer, ViTFeatureExtractor, ViTModel
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Grayscale
import numpy as np
from PIL import Image

import pickle
with open('../covid_log/covid_train.pkl', 'rb') as f:
    train_text = pickle.load(f)

with open('../covid_log/covid_test.pkl', 'rb') as f:
    test_text = pickle.load(f)


with open('../covid_log/covid_train_img.pkl', 'rb') as f:
    train_image= pickle.load(f)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained BERT
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', truncation_side='left')
model_txt = AutoModel.from_pretrained('FacebookAI/roberta-base').to(device)

# Load feature extractor and model
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)

preprocess = Compose([
        Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
    ])

def early_feature(text, image, preprocess):
    # Input text
    inputs_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

    # Input img
    inputs_img = Image.fromarray(image)
    inputs_img = preprocess(inputs_img)

    # Extract features
    with torch.no_grad():
        outputs_bert = model_txt(inputs_text['input_ids'].to(device),inputs_text['attention_mask'].to(device))
        outputs_vit = model_vit(inputs_img.unsqueeze(0).to(device))

    # Pooled output: (batch_size, hidden_size)
    pooled_output_bert = outputs_bert.pooler_output

    # Pooled output: (batch_size, hidden_size)
    pooled_output_vit = outputs_vit.pooler_output
    joint_embedding = torch.cat((pooled_output_bert, pooled_output_vit), dim=1)
    return joint_embedding.cpu().numpy()

list_feat_train = []
for x,y in zip(train_text, train_image):
       feature = early_feature(x,y,preprocess)
       list_feat_train.append(feature)

list_feat_test = []
for x,y in zip(test_text, test_image):
       feature = early_feature(x,y,preprocess)
       list_feat_test.append(feature)

       
with open('multimodal_ml/embeddings/early_train.pkl', 'wb') as handle:
       pickle.dump(list_feat_train, handle)

with open('multimodal_ml/embeddings/early_test.pkl', 'wb') as handle:
       pickle.dump(list_feat_test, handle)