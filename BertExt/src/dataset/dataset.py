import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding

class BertExt_Dataset(Dataset):

    __doc__ = r"""
        Referred to the repos below;
        https://github.com/nlpyang/PreSumm
        https://github.com/KPFBERT/kpfbertsum

        Returns:
            ids: 'id' value of the data, which is to index the document from the prediction
            encodings: input_ids, token_type_ids, attention_mask
                token_type_ids alternates between 0 and 1 to separate the sentences
            cls_token_ids: identify CLS tokens representing sentences among all tokens
            ext_label: extractive label to train in sentence-level binary classification
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            max_seq_len: int = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad = tokenizer.pad_token_id # padding token의 id


    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index: int):
        row = self.data.iloc[index]

        # load and tokenize each sentence
        encodings = []
        for sent in row['text']:
            encoding = self.tokenizer(
                sent,
                add_special_tokens=True,
            )
            encodings.append(encoding)

        input_ids, token_type_ids, attention_mask = [], [], []
        ext_label, cls_token_ids = [], []

        # seperate each of sequences
        seq_id = 0 # for segment embedding
        for enc in encodings:
            if seq_id > 1:
                seq_id = 0
            cls_token_ids += [len(input_ids)] # each [CLS] symbol collects features for the sentence preceding it.
            input_ids += enc['input_ids']
            token_type_ids += len(enc['input_ids']) * [seq_id]
            attention_mask += len(enc['input_ids']) * [1]

            if encodings.index(enc) in row['extractive']:
                ext_label += [1] # 정답 추출 문장
            else:
                ext_label += [0]

            seq_id += 1

            # truncate inputs
            if len(input_ids) == self.max_seq_len:
                break

            elif len(input_ids) > self.max_seq_len:
                sep = input_ids[-1] # sep token
                input_ids = input_ids[:self.max_seq_len - 1] + [sep]
                token_type_ids = token_type_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]
                break
        
        # pad inputs
        if len(input_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += pad_len * [self.pad]
            token_type_ids += pad_len * [0]
            attention_mask += pad_len * [0]

        # adjust for BertSum_Ext
        # 모델에 입력으로 넣기 위해 길이를 통일시킴
        if len(cls_token_ids) < self.max_seq_len:
            pad_len = self.max_seq_len - len(cls_token_ids)
            cls_token_ids += pad_len * [-1]
            ext_label += pad_len * [0]

        encodings = BatchEncoding(
            {
                'input_ids': torch.tensor(input_ids),
                'token_type_ids': torch.tensor(token_type_ids),
                'attention_mask': torch.tensor(attention_mask),
            }
        )

        return dict(
            id=row['id'],
            encodings=encodings,
            cls_token_ids=torch.tensor(cls_token_ids),
            ext_label=torch.tensor(ext_label)
        )