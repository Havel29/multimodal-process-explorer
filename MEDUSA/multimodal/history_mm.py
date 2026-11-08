from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class  MMDataset(Dataset):
    def __init__(self, texts, images, labels, tokenizer, max_len, transform):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.images = images
        self.transform = transform


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rgb_img = np.stack([self.images[idx]] * 3, axis=-1)
        image = Image.fromarray(rgb_img.astype(np.uint8))
        label = self.labels[idx]


        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
      
        inputs = self.transform(
            images=image,
            return_tensors="pt",
        )

        return {
            'text': text,
            'image': inputs['pixel_values'].squeeze(0),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }