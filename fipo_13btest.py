# fast_infer.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/tulu-2-dpo-13b"  # 7B 會更快
DTYPE = torch.bfloat16  # 沒有 bfloat16 就改 torch.float16

print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_safetensors=True,             # 優先用 safetensors，載入更穩
    torch_dtype=DTYPE,                # 明確用 BF16/FP16，避免 FP32 變慢
    device_map="auto",                # 放到 GPU
    attn_implementation="sdpa",       # 用 PyTorch 原生 SDPA，無需額外庫
)
model.eval()                          # 明確 eval 模式
torch.backends.cuda.matmul.allow_tf32 = True  # 有 Ampere+ 卡會更快

def optimize(instr: str) -> str:
    prompt = f"""<|user|>
    Rewrite the following instruction to make it more specific.
    "{instr}"
    <|assistant|>
    """
    
    # 盡量縮短輸出長度，關閉抽樣；延遲會明顯下降
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,    # ↓ 先降到 64；越小越快
            do_sample=False,      # 用貪婪解碼（最快）
            use_cache=True,       # 用 KV cache（預設開，這邊強調一下）
            pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    text = text.strip().strip('"').strip("「」").strip("“”").strip()
    # 有時模型會回多行，僅保留第一行
    text = text.splitlines()[0].strip()
    return text
    #return tok.decode(out[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    while True:
        try:
            q = input("原始指令> ").strip()
            if not q: continue
            print("優化後  >", optimize(q),"\n")
        except (EOFError, KeyboardInterrupt):
            break

# print(optimize("如何寫好論文"))