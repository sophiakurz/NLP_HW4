import torch, numpy as np
from transformers import AutoTokenizer, AutoModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

class TextEncoder:
    def __init__(self, device='cpu'):
        self.tok = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')
        self.mdl = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base').to(device)
        self.analyzer = SentimentIntensityAnalyzer()
        self.device = device

    @torch.no_grad()
    def encode_batch(self, texts, bs=64):
        out = []
        for i in tqdm(range(0, len(texts), bs), desc='encode RoBERTa'):
            batch = texts[i:i+bs]
            tok_out = self.tok(batch, padding=True, truncation=True,
                               max_length=64, return_tensors='pt').to(self.device)
            cls = self.mdl(**tok_out).last_hidden_state[:,0]  # [B,768]
            sent = torch.tensor([self.analyzer.polarity_scores(t)['compound'] for t in batch], 
                                device=self.device).unsqueeze(1)  # [B,1]
            out.append(torch.cat([cls, sent], dim=1).cpu())
        return torch.cat(out, dim=0)  # [N, 769]
