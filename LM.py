import torch.nn as nn
from transformers import AutoModel

from torch_geometric.nn.models import MLP


class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        card = './distilroberta-base'
        self.LM = AutoModel.from_pretrained(card)
        
        self.classifier = MLP(in_channels=self.LM.config.hidden_size, hidden_channels=model_config['classifier_hidden_dim'], out_channels=model_config['output_size'], num_layers=model_config['classifier_n_layers'], act=model_config['activation'])

        self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        self.LM.attention_probs_dropout_prob = model_config['att_dropout']

        self.linear_out = nn.Linear(768, model_config['classifier_hidden_dim'])

    def forward(self, tokenized_tensors):
        out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
        embedding = out[-1].mean(dim=1)
        
        # return embedding.detach(), self.classifier(embedding)

        return self.linear_out(embedding), self.classifier(embedding)