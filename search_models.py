import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

from config import config


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def re_rank(search_query, texts, docs, return_num):
    re_rank_scores = [res['score'] for res in cross_encoder([{'text': search_query, 'text_pair': text} for text in texts], max_length=512, truncation=True)]
    re_ranked_res = sorted(
        [[doc, score] for doc, score in zip(docs, re_rank_scores)],
        key=lambda x: x[-1],
        reverse=True,
    )[:return_num]

    docs = [item[0] for item in re_ranked_res]

    return docs


def get_embedding(text, max_length=512):
    with torch.no_grad():
        batch_dict = encoder_tokenizer([text], max_length=max_length, truncation=True, return_tensors='pt').to('cuda')
        outputs = encoder_model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()


encoder_tokenizer = AutoTokenizer.from_pretrained(config['encoder_model_name'])
encoder_model = AutoModel.from_pretrained(config['encoder_model_name']).to('cuda')

cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(config['cross_encoder_model_name']).to('cuda')
cross_encoder_tokenizer = AutoTokenizer.from_pretrained(config['cross_encoder_model_name'])
cross_encoder = pipeline('text-classification', model=cross_encoder_model, tokenizer=cross_encoder_tokenizer)
