from LM import LM_Model
from transformers import AutoTokenizer
from GNNs import RGCN


def build_LM_model(model_config):
    # build LM_model
    LM_model = LM_Model(model_config).to(model_config['device'])

    card = './distilroberta-base'
    LM_tokenizer = AutoTokenizer.from_pretrained(card)

    special_tokens_dict = {'additional_special_tokens': ['SEP']}
    LM_tokenizer.add_special_tokens(special_tokens_dict)
    LM_model.LM.resize_token_embeddings(len(LM_tokenizer))

    print('Information about LM model:')
    print('total params:', sum(p.numel() for p in LM_model.parameters()))
    return LM_model, LM_tokenizer


def build_GNN_model(model_config):
    # build GNN_model
    GNN_model = RGCN(model_config).to(model_config['device'])

    print('Information about GNN model:')
    print(GNN_model)
    print('total params:', sum(p.numel() for p in GNN_model.parameters()))
    
    return GNN_model