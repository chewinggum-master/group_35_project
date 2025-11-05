#!/usr/bin/env python3
"""
FIPO Instance-level Optimization for GSM8K (完全復現論文實作)

此腳本實作論文中真正使用的 instance-level optimization：
- 對每一道測試題，用 13B 優化器生成專屬的優化 prompt
- 用優化後的 prompt 讓 Llama2-7B 作答
- 完全按照論文設定，不加額外約束

輸出檔案：
- runs/gsm8k_instance_optimized_prompts_<N>.jsonl: 每題的優化 prompt
- runs/gsm8k_instance_13b_results_<N>.json: 最終評測結果（含每題的優化 prompt）
- runs/gsm8k_instance_progress_<N>.log: 進度記錄
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import os, json, argparse, re, gc
from typing import List, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional 4-bit
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# CUDA/allocator safeguards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cuda.matmul.allow_tf32 = True


def load_prompts_template() -> Dict[str, str]:
    """載入官方 FIPO optimizer template"""
    path = os.path.join("data", "prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_optimizer_meta_prompt(naive_question: str, g_n_words: int = 60) -> str:
    """為單一題目建立 optimizer 的 meta-prompt（按照論文原始設定）。
    
    使用官方 prompts.json 的 optimizer template：
    - S_P: 原始題目
    - O_C: 留空（不給範例答案，讓優化器自由發揮）
    - G_N: 字數限制
    """
    tmpl = load_prompts_template()["optimizer"]
    
    # 按照論文設定：S_P 是原始題目，O_C 空，G_N 是字數限制
    meta = (
        tmpl
        .replace("S_P", naive_question)
        .replace("O_C", "")
        .replace("G_N", str(g_n_words))
    )
    
    # Tulu chat format
    return f"<|user|>\n{meta}\n<|assistant|>\n"


@torch.inference_mode()
def generate_optimized_prompt_for_question(
    question: str,
    optimizer_model,
    optimizer_tokenizer,
    max_input_len: int = 768,
    max_new_tokens: int = 256,
) -> str:
    """為單一題目生成優化後的 prompt"""
    meta_prompt = build_optimizer_meta_prompt(question)
    
    inputs = optimizer_tokenizer(
        meta_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    ).to(optimizer_model.device)
    
    # 使用論文設定：temperature=0.8, top_p=0.95（參考官方 test_inference.sh）
    outputs = optimizer_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        pad_token_id=optimizer_tokenizer.eos_token_id,
    )
    
    text = optimizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取優化後的 prompt（移除 meta-prompt 部分）
    if "<|assistant|>" in text:
        optimized = text.split("<|assistant|>")[-1].strip()
    else:
        optimized = text.strip()
    
    # 清理可能的冗餘前綴
    optimized = optimized.replace("Golden Prompt:", "").strip()
    
    return optimized


def extract_number(s: str) -> str | None:
    """從生成文本中提取最終答案數字"""
    if not s:
        return None
    try:
        lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
        # 優先找 "Answer:" 或 "####" 標記的行
        for ln in reversed(lines[-5:]):
            if re.search(r"(?i)(answer\s*:|####)", ln):
                nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", ln)
                if nums:
                    # 移除千分位逗號
                    return nums[-1].replace(",", "")
        # 退而求其次，找最後一個數字
        m = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
        return m[-1].replace(",", "") if m else None
    except Exception:
        m = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
        return m[-1].replace(",", "") if m else None


@torch.inference_mode()
def evaluate_gsm8k_instance_level(
    optimizer_model_id: str,
    generator_model_id: str = "meta-llama/Llama-2-7b-hf",
    use_4bit: bool = True,
    limit: int | None = None,
    out_prefix: str = "gsm8k_instance",
):
    """
    Instance-level FIPO evaluation:
    1. 載入 optimizer (13B) 和 generator (7B)
    2. 對每題測試題：
       a. 用 optimizer 生成優化 prompt
       b. 用優化 prompt 讓 generator 作答
       c. 提取答案並比對
    3. 儲存所有優化 prompts 和結果
    """
    print("\n" + "=" * 70)
    print("Instance-level FIPO Evaluation for GSM8K")
    print(f"Optimizer: {optimizer_model_id}")
    print(f"Generator: {generator_model_id}")
    print("=" * 70 + "\n")
    
    # ========== 載入 Optimizer ==========
    print(f"[1/3] Loading Optimizer: {optimizer_model_id}", flush=True)
    torch.cuda.empty_cache(); gc.collect()
    
    opt_tok = AutoTokenizer.from_pretrained(optimizer_model_id, use_fast=True)
    if use_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes not available; install it or disable --optimizer_4bit")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        opt_model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            quantization_config=bnb,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        opt_model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    opt_model.eval()
    print("Optimizer loaded.\n", flush=True)
    
    # ========== 載入 Generator ==========
    print(f"[2/3] Loading Generator: {generator_model_id}", flush=True)
    # 如果顯存不夠，這裡可能需要先卸載 optimizer 或用 CPU offload
    # 為簡化實作，假設顯存足夠同時載入 13B(4bit) + 7B(fp16)
    
    gen_tok = AutoTokenizer.from_pretrained(generator_model_id, use_fast=True)
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        generator_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    gen_model.eval()
    print("Generator loaded.\n", flush=True)
    
    # ========== 載入測試集 ==========
    print(f"[3/3] Loading GSM8K test set", flush=True)
    ds = load_dataset("gsm8k", "main", split="test")
    if limit is not None:
        ds = ds.select(range(limit))
    total = len(ds)
    print(f"Dataset size: {total}\n", flush=True)
    
    # ========== 準備輸出檔案 ==========
    os.makedirs("runs", exist_ok=True)
    optimized_prompts_file = os.path.join("runs", f"{out_prefix}_optimized_prompts_{total}.jsonl")
    progress_log = os.path.join("runs", f"{out_prefix}_progress_{total}.log")
    full_log = os.path.join("runs", f"{out_prefix}_full_{total}.log")
    result_json = os.path.join("runs", f"{out_prefix}_13b_results_{total}.json")
    
    print(f"Optimized prompts: {optimized_prompts_file}")
    print(f"Progress log     : {progress_log}")
    print(f"Full log         : {full_log}")
    print(f"Final results    : {result_json}\n")
    
    # 清空或建立檔案
    with open(progress_log, "w", encoding="utf-8") as pf:
        pf.write(f"Instance-level FIPO on {total} questions\n")
    
    with open(full_log, "w", encoding="utf-8") as fl:
        fl.write(f"Instance-level FIPO Evaluation Log\n")
        fl.write(f"Optimizer: {optimizer_model_id}\n")
        fl.write(f"Generator: {generator_model_id}\n")
        fl.write(f"Total questions: {total}\n")
        fl.write("=" * 70 + "\n\n")
    
    open(optimized_prompts_file, "w", encoding="utf-8").close()
    
    # ========== 逐題優化與評測 ==========
    print("=" * 70)
    print("Starting instance-level optimization and evaluation...")
    print("=" * 70 + "\n")
    
    correct = 0
    predictions = []
    
    for i, item in enumerate(ds):
        q, ans = item["question"], item["answer"]
        gold = extract_number(ans)
        
        # Step 1: 用 optimizer 生成該題的優化 prompt（按照論文原始設定）
        try:
            optimized_prompt = generate_optimized_prompt_for_question(
                question=q,
                optimizer_model=opt_model,
                optimizer_tokenizer=opt_tok,
            )
        except Exception as e:
            msg = f"[Q{i}] Optimizer failed: {type(e).__name__}: {str(e)[:100]}"
            print(msg)
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
            optimized_prompt = q  # Fallback to naive question
        
        # 記錄優化後的 prompt
        with open(optimized_prompts_file, "a", encoding="utf-8") as opf:
            opf.write(json.dumps({
                "idx": i,
                "original_question": q,
                "optimized_prompt": optimized_prompt,
            }, ensure_ascii=False) + "\n")
        
        # Step 2: 用優化後的 prompt 讓 generator 作答
        # 論文中，優化後的 prompt 就是完整輸入，通常不再加 few-shot
        # （few-shot 是在訓練 optimizer 時用的，不是給下游 generator 用的）
        try:
            inputs = gen_tok(
                optimized_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(gen_model.device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # Greedy for stability
                    pad_token_id=gen_tok.eos_token_id,
                )
            
            generated = gen_tok.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            msg = f"[Q{i}] Generator failed: {type(e).__name__}: {str(e)[:100]}"
            print(msg)
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
            generated = f"<GEN_ERROR: {type(e).__name__}>"
        
        # Step 3: 提取答案並評分
        pred = extract_number(generated)
        ok = int(pred == gold)
        correct += ok
        
        rec = {
            "idx": i,
            "original_question": q,
            "optimized_prompt": optimized_prompt,
            "generated": generated,
            "pred_num": pred,
            "gold": gold,
            "correct": ok,
        }
        predictions.append(rec)
        
        # 每 10 題報告進度
        if (i + 1) % 10 == 0:
            acc_now = correct / (i + 1)
            msg = f"[{i+1}/{total}] Acc: {acc_now*100:.2f}% ({correct}/{i+1})"
            print(msg, flush=True)
            with open(progress_log, "a", encoding="utf-8") as pf:
                pf.write(f"step {i+1}/{total} acc={acc_now:.4f}\n")
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
        
        # 定期清理顯存
        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()
    
    # ========== 儲存最終結果 ==========
    acc = correct / total if total else 0.0
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "method": "instance-level FIPO",
            "optimizer": optimizer_model_id,
            "generator": generator_model_id,
            "optimizer_4bit": use_4bit,
            "total_questions": total,
            "correct": correct,
            "accuracy": acc,
            "predictions": predictions,
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Final Accuracy: {acc*100:.2f}% ({correct}/{total})")
    print(f"Results saved to: {result_json}")
    print(f"Optimized prompts: {optimized_prompts_file}")
    print("=" * 70 + "\n")
    
    # 清理
    del opt_model, gen_model, opt_tok, gen_tok
    torch.cuda.empty_cache()
    gc.collect()
    
    return acc, result_json


def main():
    parser = argparse.ArgumentParser(
        description="Instance-level FIPO for GSM8K (按照論文實作)"
    )
    parser.add_argument("--optimizer_model", type=str, default="allenai/tulu-2-dpo-13b")
    parser.add_argument("--generator_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--optimizer_4bit", action="store_true", help="Use 4-bit quantization for optimizer (recommended)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test questions (for debugging)")
    parser.add_argument("--out_prefix", type=str, default="gsm8k_instance", help="Output file prefix")
    parser.add_argument("--mode", type=str, choices=["optimize", "generate", "full"], default="full",
                        help="optimize: 只生成優化 prompt | generate: 用已有 prompt 生成答案 | full: 一次完成兩階段(需要足夠記憶體)")
    parser.add_argument("--optimized_prompts_file", type=str, default=None,
                        help="For generate mode: path to optimized prompts JSONL file")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print("GPU cache reset\n")
    
    evaluate_gsm8k_instance_level(
        optimizer_model_id=args.optimizer_model,
        generator_model_id=args.generator_model,
        use_4bit=args.optimizer_4bit,
        limit=args.limit,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
