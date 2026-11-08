import torch.nn as nn
import torch

class ConcatFusion(nn.Module):
    def __init__(self, bert_encoder, vit_encoder, num_labels, hidden_dim=512, dropout_rate=0.1):
        super(ConcatFusion, self).__init__()
        self.txt_encoder = bert_encoder
        self.vis_encoder = vit_encoder
        self.combined_size = self.txt_encoder.config.hidden_size + self.vis_encoder.config.hidden_size
        
        # MLP finale con più layer
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, image):
        txt_output = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)
        img_output = self.vis_encoder(image)
        txt_feature = txt_output.pooler_output
        img_feature = img_output.pooler_output
        features = torch.cat((txt_feature, img_feature), 1)
        logits = self.mlp(features)
        return logits
