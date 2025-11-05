import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#model_id = "allenai/tulu-2-7b"

model_id = "allenai/tulu-2-dpo-70b"

print(f"Loading model: {model_id}")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 或 torch.float16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",     # 讓它自己把部分放到 CPU
    trust_remote_code=True   
)


prompt = """<|user|>
Rewrite the following instruction to make it more specific:
"如何寫出好的論文?"
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("=== Generated ===")
print(text)