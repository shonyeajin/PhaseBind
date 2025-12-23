import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import sys
import argparse
from typing import List, Tuple
import numpy as np
import torch


ENCODER_HALF = "Rostlab/prot_t5_xl_half_uniref50-enc"
FULL_MODEL  = "Rostlab/prot_t5_xl_uniref50"


seq_num=1


def load_sequences_from_txt(path: str) -> Tuple[List[str], List[str]]:
    ids, seqs = [],[]
    with open(path, 'r') as f:
        for i,line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            ids.append(f'seq_{i}')
            seqs.append(s)
    return ids, seqs


def preprocess_sequences(seqs: List[str]) -> List[str]:
    """ProtT5 권장 전처리: U/Z/O/B -> X, 공백 토크나이징용 스페이스 삽입"""
    out = []
    for s in seqs:
        s = s.upper()
        s = re.sub(r"[UZOB]", "X", s)
        out.append(" ".join(list(s)))
    return out



def get_model(model_name:str, device:str):
    from transformers import T5Tokenizer, T5Model, T5EncoderModel
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    if 'enc' in model_name:
        model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16 if 'half' in model_name else None)
    else:
        model = T5Model.from_pretrained(model_name)

    if device =='cuda':
        model = model.to(device)
    
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_batch(tokenizer, model, batch_seqs: List[str], device: str):
    global seq_num
    ids = tokenizer.batch_encode_plus(
        batch_seqs, add_special_tokens=True, padding=True, return_tensors='pt'
    )
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    print(f'{seq_num}--->{input_ids.shape}')
    seq_num+=1
    outputs = model(input_ids= input_ids, attention_mask=attention_mask)
    reps = outputs.last_hidden_state
    return reps,attention_mask




def main():
    ap = argparse.ArgumentParser()
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=str, default=None, help="한 줄 한 시퀀스 TXT 경로")
    ap.add_argument("--out", type=str, default="prot_t5_embeddings.npz", help="출력 npz 파일명")
    ap.add_argument("--model", type=str, default=ENCODER_HALF, help=f"모델 이름 (기본: {ENCODER_HALF}, 대안: {FULL_MODEL})")
    ap.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    args = ap.parse_args()

    if args.txt:
        ids, seqs = load_sequences_from_txt(args.txt)
    else:
        print('txt파일 입력하세요')
        ids = ['demo1', 'demo2']
        seqs = ['PRTEINO', "SEQQWENCE"]
    
    # print(f'ids:{ids}')
    # print(f'seqs:{seqs}')

    proc = preprocess_sequences(seqs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Info] device={device}')

    tokenizer, model = get_model(args.model, device)
    hidden_dim = None

    per_protein = []
    per_residue = [] # 가변 길이라 리스트로 보관 (npz로 object 저장)

    # 배치 처리
    for i in range(0, len(proc), args.batch_size):
        batch_ids = ids[i:i+args.batch_size]
        batch_seqs = proc[i:i+args.batch_size]

        reps, attn = embed_batch(tokenizer, model, batch_seqs, device) # (B, L, H)
        reps = reps.float().cpu()
        attn = attn.cpu()

        if hidden_dim is None:
            hidden_dim = reps.shape[-1]
            print(f'[Info] embedding dim = {hidden_dim}')
        
        # 패딩 제거 후 평균(per-protein), 패딩 제거한 per-residue 보관
        for b in range(reps.size(0)):
            length = int(attn[b].sum().item())
            # special tokens까지 포함될 수 있으므로 실제 AA 토큰만 쓰고 싶으면 아래 offset 조정 가능
            # ProtT5 토큰화는 [BOS/EOS] 등이 포함 → 여기서는 단순히 attn==1인 토큰들 기준
            emb_tokens = reps[b, :length-1, :]  # (len, H)
            per_residue.append(emb_tokens.numpy())         # variable length
            per_protein.append(emb_tokens.mean(dim=0).numpy())  # (H,)        

    per_protein = np.stack(per_protein, axis=0) # (N, H)

    # ID/원본 시퀀스도 함께 저장
    out_path = args.out
    np.savez(
        out_path,
        ids=np.array(ids, dtype=object),
        seqs=np.array(seqs, dtype=object),
        per_protein=per_protein,             # (N, H)
        per_residue=np.array(per_residue, dtype=object),  # list of (L_i, H)
        model=np.array(args.model),
        dim=np.array(hidden_dim),
    )
    print(f"[Done] saved → {out_path}")

if __name__ == "__main__":
    main()

