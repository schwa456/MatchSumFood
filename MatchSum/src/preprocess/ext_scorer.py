from typing import Optional
import torch
from .bertext import BertExt

def load_scorer_model(
    tokenizer,
    encoder_path: Optional[str],
    enc_num_layers: Optional[int],
    enc_intermediate_size: Optional[int],
    enc_num_attention_heads: Optional[int],
    enc_dropout_prob: Optional[float],
    device='cuda'
    ):
    model = BertExt(
        base_checkpoint=tokenizer,
        enc_num_layers=enc_num_layers,
        enc_intermediate_size=enc_intermediate_size,
        enc_num_attention_heads=enc_num_attention_heads,
        enc_dropout_prob=enc_dropout_prob
    ).to(device)
    
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    model.eval()
    
    return model

@torch.no_grad()
def get_salience_score(model, tokenizer, sentences, device='cuda'):
    model.eval()

    # 문장별로 토크나이즈하고 CLS 위치 추적
    input_ids_list = []
    cls_token_positions = []

    for sent in sentences:
        tokens = tokenizer.encode(
            sent,
            add_special_tokens=True,
            max_length=128,
            truncation=True
        )  # [CLS] ... [SEP]
        cls_pos = len(input_ids_list)
        cls_token_positions.append(cls_pos)
        input_ids_list.extend(tokens)

    # max length 제한
    input_ids_list = input_ids_list[:512]
    input_ids = torch.tensor([input_ids_list], device=device)  # (1, L)
    attention_mask = torch.ones_like(input_ids, device=device)  # 마스크 (1, L)

    # 유효한 CLS 위치만 필터링
    cls_token_ids = [pos for pos in cls_token_positions if pos < input_ids.size(1)]
    if len(cls_token_ids) == 0:
        print(f"[ERROR] No valid CLS positions under max length 512.")
        return []

    cls_token_ids = torch.tensor([cls_token_ids], device=device)  # (1, N)

    try:
        # BatchEncoding 객체로 전달
        encodings = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        outputs = model(encodings, cls_token_ids)
        scores = [float(s) for s in outputs['confidence'][0]]
        return scores
    except Exception as e:
        print(f"[ERROR] Scoring Failed: {e}")
        return []


if __name__ == '__main__':
    """
    sentences = ['[1] 질서위반행위규제법은 과태료의 부과대상인 질서위반행위에 대하여도 책임주의 원칙을 채택하여 제7조에서 "고의 또는 과실이 없는 질서위반행위는 과태료를 부과하지 아니한다."고 규정하고 있으므로,', 
                 '질서위반행위를 한 자가 자신의 책임 없는 사유로 위반행위에 이르렀다고 주장하는 경우 법원으로서는 그 내용을 살펴 행위자에게 고의나 과실이 있는지를 따져보아야 한다.', 
                 '[2] 주거용으로 토지거래허가를 받아 매수한 토지지분을 허가받은 목적대로 이용하지 않은 채 방치하였다는 이유로 과태료 처분을 받은 자가 토지거래허가를 받은 직후 주거용 건물을 신축하려고 하였으나', 
                 '허가 당시 도로사용승낙을 하여 주었던 인근 토지 소유자가 태도를 바꿔 차량 출입을 방해함으로써 착공에 이르지 못하였을 뿐이므로 자신에게 책임이 없다고 주장한 사안에서,', 
                 '이는 제3자의 방해로 토지를 이용할 수 없었을 뿐 의도적으로 허가 목적에 따른 이용을 회피한 것이 아니라는 취지로 질서위반행위규제법 제7조에 따라 고의나 과실을 부인하는 것으로 이해될 여지가 있음에도,', 
                 '위 주장이 과태료 부과에 아무런 장애가 되지 않는다고 보아 이에 관한 심리와 판단에 나아가지 않은 채 위 주장을 배척한 원심판단에는 법리오해의 위법이 있다고 한 사례.']
    
    sentences = ['헌법 제28조는 "형사피의자 또는 형사피고인으로서 구금되었던 자가 법률이 정하는 불기소처분을 받거나 무죄판결을 받은 때에는 법률이 정하는 바에 의하여 국가에 정당한 보상을 청구할 수 있다."고 규정하고,', 
                 '형사보상 및 명예회복에 관한 법률(이하 \'형사보상법\'이라 한다) 제2조 제1항은 ""형사소송법에 따른 일반 절차 또는 재심이나 비상상고 절차에서 무죄재판을 받아 확정된 사건의 피고인이 미결구금을 당하였을 때에는 이 법에 따라 국가에 대하여 그 구금에 대한 보상을 청구할 수 있다.""고 규정하고 있다.', 
                 '이와 같은 형사보상법 조항은 입법 취지와 목적 및 내용 등에 비추어 재판에 의하여 무죄의 판단을 받은 자가 재판에 이르기까지 억울하게 미결구금을 당한 경우 보상을 청구할 수 있도록 하기 위한 것이므로,', 
                 '판결 주문에서 무죄가 선고된 경우뿐만 아니라 판결 이유에서 무죄로 판단된 경우에도 미결구금 가운데 무죄로 판단된 부분의 수사와 심리에 필요하였다고 인정된 부분에 관하여는 보상을 청구할 수 있고,', 
                 '다만 형사보상법 제4조 제3호를 유추적용하여 법원의 재량으로 보상청구의 전부 또는 일부를 기각할 수 있을 뿐이다.']
    """
    sentences = ['가. 행정처분의 취소를 구하는 소송에 있어서는,', 
                 '실질적 법치주의와 행정처분의 상대방인 국민에 대한 신뢰보호라는 견지에서,', 
                 '처분청은 당초의 처분사유와 기본적 사실관계에 있어서 동일성이 인정되는 한도 내에서만 새로운 처분사유를 추가하거나 변경할 수 있고,', 
                 '기본적 사실관계와 동일성이 전혀 없는 별개의 사실을 들어 처분사유로서 주장함은 허용되지 아니하며,', 
                 '법원으로서도 당초 처분사유와 기본적 사실관계의 동일성이 없는 사실은 이를 처분사유로서 인정할 수 없다.', 
                 '나. 피고가 당초 처분사유로 삼은 구 자동차운수사업법(1994.8.3. 법률 제4780호로 개정되기 전의 것) 제6조 제1항 제3호 소정의 요건을 충족하지 못한다는 사유와 원심이 그 처분사유로 인정한 같은 법 제6조 제1항 제4호 소정의 요건을 충족하지 못한다는 사유는 그 기본적 사실관계가 동일하다고 볼 수 없다고 한 사례.']

    from transformers import AutoTokenizer
    
    print('Importing Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    print('Importing Tokenizer completed.')
    
    print('Importing Encoder Model...')
    encoder = load_scorer_model(
        tokenizer='klue/bert-base', 
        encoder_path='/home/food/people/hyeonjin/FoodSafetyCrawler/matchsum/BertExt/bertext_only.pt',
        enc_num_layers=2,
        enc_intermediate_size=2048,
        enc_num_attention_heads=8,
        enc_dropout_prob=0.1,
        device='cuda'
    )
    print('Importing Encoder Model completed.')
    
    print('Sentences:')
    for sent in sentences:
        print(sent)
    print(' ')
    
    scores = get_salience_score(encoder, tokenizer, sentences, device='cuda')
    
    for sent, score in zip(sentences, scores):
        print(f"Score: {score:.4f} | {sent}")
