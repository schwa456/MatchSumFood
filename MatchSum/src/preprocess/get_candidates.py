from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from itertools import combinations
import torch, json, os
from tqdm import tqdm
from src.score.rouge_score import RougeScorer

from .ext_scorer import load_scorer_model, get_salience_score

def generate_candidate_summaries(
    doc_sents, model, tokenizer, device='cuda', topk=5, sel_lens=[2, 3]
):
    scores = get_salience_score(model, tokenizer, doc_sents, device=device)
    topk_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    topk_indices = sorted(topk_indices)
    
    candidates = []
    for sel in sel_lens:
        for combo in combinations(topk_indices, sel):
            summary = ' '.join([doc_sents[i] for i in combo])
            candidates.append({
                'indices': list(combo),
                'text': summary,
            })
    
    return candidates

def process_dataset(
    input_jsonl, output_jsonl, bert_pt_path, device='cuda', topk=5, sel_lens=[2, 3]
):
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    model = load_scorer_model(bert_pt_path, device=device)
    
    rouge = RougeScorer(['rougeL'], use_stemmer=True)
    
    with open(input_jsonl) as f:
        lines = [json.load(l) for l in f]
    
    with open(output_jsonl, 'w') as fout:
        for i, item in enumerate(tqdm(lines)):
            doc_sents = item['documents']
            summary = item.get('summary', '')
            
            candidates = generate_candidate_summaries(doc_sents, model, tokenizer, device, topk, sel_lens)
            for j, cand in enumerate(candidates):
                cand_text = cand['text']
                label = rouge.score(summary, cand_text)['rougeL'].fmeasure
                
                fout.write(json.dumps({
                    'text_id': i,
                    'candidate_id': j,
                    'text': ' '.join(doc_sents),
                    'summary': summary,
                    'candidate': cand['text'],
                    'indices': cand['indices'],
                    'label': label
                }) + '\n')