from torch.utils.data import Dataset
from collections import defaultdict
from transformers import PreTrainedTokenizer
import json, torch, ast

class MatchSumDataset(Dataset):
    def __init__(self, df, tokenizer: PreTrainedTokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Document 토크나이징
        doc_sents = row['text'] if isinstance(row['text'], list) else ast.literal_eval(row['text'])
        doc_text = ' '.join(doc_sents)
        doc_inputs = self.tokenizer(doc_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        # 2. 모든 후보 요약문 토크나이징
        candidates = ast.literal_eval(row['candidates'])
        cand_texts = [' '.join(c) for c in candidates]
        cand_inputs = self.tokenizer(cand_texts, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        # 3. 정답 요약문 (gold summary) 토크나이징
        gold_summary_sents = row['gold_summary'] if isinstance(row['gold_summary'], list) else ast.literal_eval(row['gold_summary'])
        gold_summary_text = ' '.join(gold_summary_sents)
        summary_inputs = self.tokenizer(gold_summary_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        # 4. 정답 인덱스(label)
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'doc_input_ids': doc_inputs['input_ids'].squeeze(0),
            'doc_attention_mask': doc_inputs['attention_mask'].squeeze(0),
            'cand_input_ids': cand_inputs['input_ids'], # (num_candidates, max_len)
            'cand_attention_mask': cand_inputs['attention_mask'],
            'summary_input_ids': summary_inputs['input_ids'].squeeze(0),
            'summary_attention_mask': summary_inputs['attention_mask'].squeeze(0),
            'label': label
        }
    
if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    train_df = pd.read_csv('/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/MatchSumRevised/data/train_df.csv')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    
    candidates = train_df.loc[0, 'candidates']
    candidates_list = ast.literal_eval(candidates)
    print(f"Number of candidates for doc 0: {len(candidates_list)}")
    
    train_dataset = MatchSumDataset(train_df, tokenizer)
    
    print(train_dataset[0])
    print(train_dataset[1])
    
    for i in range(10):
        print(train_dataset[i])
    
    print(len(train_df))