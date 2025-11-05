import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/tulu-2-dpo-7b"

print(f"Loading model: {model_id}")
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_safetensors=True,         
    torch_dtype="auto",
    device_map="auto",
)

def optimize(instr: str) -> str:
    prompt = f"""<|user|>
Rewrite the following instruction to make it more specific.
"{instr}"
<|assistant|>
"""
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).strip()

# if __name__ == "__main__":
#     while True:
#         try:
#             q = input("原始指令> ").strip()
#             if not q: 
#                 continue
#             print("優化後  >", optimize(q))
#         except (EOFError, KeyboardInterrupt):
#             break
print( optimize("如何寫好論文"))