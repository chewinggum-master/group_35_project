#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instance-level FIPO for GSM8K - 兩階段執行版本（解決 OOM 問題）

參考論文實作，分為兩個獨立階段：
1. optimize: 只載入 optimizer，為每題生成優化 prompt 並存檔
2. generate: 只載入 generator，用優化後的 prompt 進行推理

這樣可以避免同時載入兩個大模型造成的 OOM 問題。

修正重點：
- Optimizer max_new_tokens 從 128 提升到 256（避免截斷）
- 清理 **Golden Prompt:** 前綴
- 空/短 prompt 自動 fallback 到 silver_prompt
- Generator 加 repetition_penalty=1.1 避免瘋狂重複
"""

import os
import sys
import gc
import re
import json
import argparse
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset

# CUDA/allocator safeguards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cuda.matmul.allow_tf32 = True


def extract_number(text: str) -> Optional[float]:
    """從 GSM8K 答案中提取最終數字"""
    text = str(text)
    parts = text.split("####")
    if len(parts) > 1:
        ans_str = parts[-1].strip()
    else:
        ans_str = text.strip()
    
    ans_str = ans_str.replace(",", "")
    match = re.search(r"-?\d+\.?\d*", ans_str)
    if match:
        try:
            return float(match.group())
        except:
            return None
    return None


def load_prompts_template():
    """載入官方 prompts.json 模板"""
    template_path = "data/prompts.json"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"需要 {template_path}，請確認檔案存在")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_silver_prompt_for_gsm8k(question: str) -> str:
    """建立 GSM8K 用的 baseline（Silver）prompt，內含題目本身。
    我們讓 optimizer 以此為基礎產生更好的 Golden Prompt。
    """
    return (
        "You are a helpful math reasoning assistant.\n"
        "Solve the following math word problem step by step, then on the last line write '#### ' followed by the final numeric answer.\n\n"
        "Problem:\n"
        f"{question}\n\n"
        "Answer:"
    )


def build_optimizer_meta_prompt(question: str, g_n: int = 60) -> str:
    """
    構建 meta-prompt 讓 optimizer 生成優化後的 prompt。
    完全復現論文實作，不加入任何約束。
    
    Args:
        question: GSM8K 問題
        g_n: 優化 prompt 的目標長度 (論文設為 60)
    
    Returns:
        完整的 meta-prompt
    """
    templates = load_prompts_template()
    optimizer_template = templates["optimizer"]
    # 使用正確的 placeholder：S_P, O_C, G_N
    silver = make_silver_prompt_for_gsm8k(question)
    meta_prompt = optimizer_template
    meta_prompt = meta_prompt.replace("S_P", silver)
    meta_prompt = meta_prompt.replace("O_C", "")
    meta_prompt = meta_prompt.replace("G_N", str(g_n))
    
    return meta_prompt


def generate_optimized_prompt_for_question(
    question: str,
    optimizer_model,
    optimizer_tokenizer,
) -> str:
    """
    用 optimizer 為單一問題生成優化後的 prompt
    
    Args:
        question: GSM8K 問題
        optimizer_model: Optimizer 模型
        optimizer_tokenizer: Optimizer tokenizer
    
    Returns:
        優化後的 prompt
    """
    meta_prompt = build_optimizer_meta_prompt(question, g_n=60)
    
    inputs = optimizer_tokenizer(
        meta_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(optimizer_model.device)
    
    # 論文設定：temperature=0.8, top_p=0.95
    # 提升 max_new_tokens 到 256 避免截斷
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = optimizer_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=optimizer_tokenizer.eos_token_id,
        )
    
    # 只取新生成的部分並做清理
    optimized_prompt = optimizer_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # 清理常見前綴/包裹（包含 markdown bold）
    if optimized_prompt.startswith("**Golden Prompt:**"):
        optimized_prompt = optimized_prompt[len("**Golden Prompt:**"):].strip()
    if optimized_prompt.lower().startswith("golden prompt:"):
        optimized_prompt = optimized_prompt[len("golden prompt:"):].strip()
    # 去除成對的反引號包裹
    if optimized_prompt.startswith("```") and optimized_prompt.endswith("```"):
        optimized_prompt = optimized_prompt[3:-3].strip()
    # 去除開頭的換行
    optimized_prompt = optimized_prompt.lstrip("\n")
    
    return optimized_prompt


def optimize_prompts_only(
    optimizer_model_id: str = "allenai/tulu-2-dpo-13b",
    use_4bit: bool = True,
    limit: Optional[int] = None,
    out_prefix: str = "gsm8k_instance",
    log_dir: str = "runs",
):
    """
    第一階段：只載入 optimizer，為每題生成優化 prompt
    
    Args:
        optimizer_model_id: Optimizer 模型 ID
        use_4bit: 是否使用 4-bit 量化
        limit: 限制題數（用於測試）
        out_prefix: 輸出檔案前綴
    """
    print("="*70)
    print(f"階段 1/2: 生成優化 Prompts")
    print(f"Optimizer: {optimizer_model_id}")
    print(f"4-bit 量化: {use_4bit}")
    print("="*70)
    print()
    
    # 載入 optimizer
    print(f"[1/2] Loading Optimizer: {optimizer_model_id}")
    opt_tok = AutoTokenizer.from_pretrained(optimizer_model_id, use_fast=True)
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        opt_model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    else:
        opt_model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    print("Optimizer loaded.\n")
    
    # 載入 GSM8K
    print("[2/2] Loading GSM8K test set")
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    total = len(ds)
    print(f"Dataset size: {total}\n")
    
    # 準備輸出檔案
    os.makedirs(log_dir, exist_ok=True)
    optimized_prompts_file = os.path.join(log_dir, f"{out_prefix}_optimized_prompts_{total}.jsonl")
    progress_log = os.path.join(log_dir, f"{out_prefix}_optimize_progress_{total}.log")
    full_log = os.path.join(log_dir, f"{out_prefix}_optimize_full_{total}.log")
    
    print(f"Optimized prompts: {optimized_prompts_file}")
    print(f"Progress log     : {progress_log}")
    print(f"Full log         : {full_log}")
    print()
    
    # 清空舊檔案
    for f in [optimized_prompts_file, progress_log, full_log]:
        if os.path.exists(f):
            os.remove(f)

    # full log header
    with open(full_log, "w", encoding="utf-8") as fl:
        fl.write("="*70 + "\n")
        fl.write("Stage 1: Optimize prompts\n")
        fl.write(f"Optimizer: {optimizer_model_id}\n")
        fl.write(f"4bit: {use_4bit}\n")
        fl.write(f"Total: {total}\n")
        fl.write("="*70 + "\n\n")
    
    print("="*70)
    print("開始生成優化 prompts...")
    print("="*70)
    print()
    
    # 為每題生成優化 prompt
    for i, item in enumerate(ds):
        q = item["question"]
        
        try:
            optimized_prompt = generate_optimized_prompt_for_question(
                question=q,
                optimizer_model=opt_model,
                optimizer_tokenizer=opt_tok,
            )
            # Fallback: 若優化後為空或太短，使用 silver prompt
            if not optimized_prompt or len(optimized_prompt) < 10:
                msg_warn = f"[Q{i}] Optimized prompt too short/empty, using silver prompt"
                print(msg_warn)
                with open(full_log, "a", encoding="utf-8") as fl:
                    fl.write(msg_warn + "\n")
                optimized_prompt = make_silver_prompt_for_gsm8k(q)
        except Exception as e:
            msg = f"[Q{i}] Optimizer failed: {type(e).__name__}: {str(e)[:100]}"
            print(msg)
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
            optimized_prompt = make_silver_prompt_for_gsm8k(q)  # Fallback
        
        # 儲存
        with open(optimized_prompts_file, "a", encoding="utf-8") as opf:
            opf.write(json.dumps({
                "idx": i,
                "original_question": q,
                "gold_answer": item["answer"],
                "silver_prompt": make_silver_prompt_for_gsm8k(q),
                "optimized_prompt": optimized_prompt,
            }, ensure_ascii=False) + "\n")
        
        # 進度報告
        if (i + 1) % 10 == 0:
            msg = f"[{i+1}/{total}] Generated optimized prompts"
            print(msg, flush=True)
            with open(progress_log, "a", encoding="utf-8") as pf:
                pf.write(f"{msg}\n")
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
        
        # 定期清理
        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()
    
    print()
    print("="*70)
    print(f"階段 1 完成！優化 prompts 已存至: {optimized_prompts_file}")
    print("="*70)
    print()
    with open(full_log, "a", encoding="utf-8") as fl:
        fl.write("\n" + "="*70 + "\n")
        fl.write(f"Stage 1 done. Prompts at: {optimized_prompts_file}\n")
        fl.write("="*70 + "\n")
    print("下一步：執行階段 2 (生成答案)")
    print(f"  python fipo_gsm8k_13b_2stage_fixed.py --mode generate \\")
    print(f"    --optimized_prompts_file {optimized_prompts_file}")
    print()
    
    # 清理記憶體
    del opt_model, opt_tok
    torch.cuda.empty_cache()
    gc.collect()


def generate_with_optimized_prompts(
    generator_model_id: str = "meta-llama/Llama-2-7b-hf",
    optimized_prompts_file: str = None,
    limit: Optional[int] = None,
    out_prefix: str = "gsm8k_instance",
    log_dir: str = "runs",
    gen_8bit: bool = False,
    gen_4bit: bool = False,
):
    """
    第二階段：用優化後的 prompts 進行推理
    
    Args:
        generator_model_id: Generator 模型 ID
        optimized_prompts_file: 階段 1 產生的優化 prompts JSONL 檔案
        limit: 限制題數（用於測試）
        out_prefix: 輸出檔案前綴
    """
    print("="*70)
    print(f"階段 2/2: 用優化 Prompts 生成答案")
    print(f"Generator: {generator_model_id}")
    print(f"Prompts file: {optimized_prompts_file}")
    print("="*70)
    print()
    
    # 讀取優化後的 prompts
    print("[1/2] Loading optimized prompts")
    optimized_data = []
    with open(optimized_prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            optimized_data.append(json.loads(line))
    
    if limit:
        optimized_data = optimized_data[:limit]
    total = len(optimized_data)
    print(f"Loaded {total} optimized prompts\n")
    
    # 載入 generator
    print(f"[2/2] Loading Generator: {generator_model_id}")
    gen_tok = AutoTokenizer.from_pretrained(generator_model_id, use_fast=True)
    if gen_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        print("Generator loaded in 4-bit mode.\n")
    elif gen_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        print("Generator loaded in 8-bit mode.\n")
    else:
        gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    print("Generator loaded.\n")
    
    # 準備輸出檔案
    os.makedirs(log_dir, exist_ok=True)
    results_file = os.path.join(log_dir, f"{out_prefix}_results_{total}.json")
    progress_log = os.path.join(log_dir, f"{out_prefix}_generate_progress_{total}.log")
    full_log = os.path.join(log_dir, f"{out_prefix}_generate_full_{total}.log")
    
    print(f"Results file : {results_file}")
    print(f"Progress log : {progress_log}")
    print(f"Full log     : {full_log}")
    print()
    
    # 清空舊檔案
    for f in [progress_log, full_log]:
        if os.path.exists(f):
            os.remove(f)

    # full log header
    with open(full_log, "w", encoding="utf-8") as fl:
        fl.write("="*70 + "\n")
        fl.write("Stage 2: Generate answers with optimized prompts\n")
        fl.write(f"Generator: {generator_model_id}\n")
        fl.write(f"Prompts file: {optimized_prompts_file}\n")
        fl.write(f"Total: {total}\n")
        fl.write("="*70 + "\n\n")
    
    print("="*70)
    print("開始生成答案...")
    print("="*70)
    print()
    
    # 推理
    predictions = []
    correct = 0
    
    for i, item in enumerate(optimized_data):
        optimized_prompt = item["optimized_prompt"]
        gold = extract_number(item["gold_answer"])
        
        # Fallback: 若為空，嘗試用 silver_prompt
        if not optimized_prompt or len(optimized_prompt.strip()) == 0:
            if "silver_prompt" in item:
                optimized_prompt = item["silver_prompt"]
            else:
                optimized_prompt = make_silver_prompt_for_gsm8k(item["original_question"])
        
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
                    do_sample=False,  # Greedy
                    repetition_penalty=1.1,  # 避免瘋狂重複
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
        
        # 評分
        pred = extract_number(generated)
        ok = int(pred == gold)
        correct += ok
        
        rec = {
            "idx": i,
            "original_question": item["original_question"],
            "optimized_prompt": optimized_prompt,
            "generated": generated,
            "pred_num": pred,
            "gold": gold,
            "correct": ok,
        }
        predictions.append(rec)
        
        # 進度報告
        if (i + 1) % 10 == 0:
            acc_now = correct / (i + 1)
            msg = f"[{i+1}/{total}] Acc: {acc_now*100:.2f}% ({correct}/{i+1})"
            print(msg, flush=True)
            with open(progress_log, "a", encoding="utf-8") as pf:
                pf.write(f"{msg}\n")
            with open(full_log, "a", encoding="utf-8") as fl:
                fl.write(msg + "\n")
        
        # 定期清理
        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()
    
    # 最終結果
    final_acc = correct / total
    
    result_summary = {
        "generator_model": generator_model_id,
        "total": total,
        "correct": correct,
        "accuracy": final_acc,
        "predictions": predictions,
    }
    
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(result_summary, rf, ensure_ascii=False, indent=2)
    
    print()
    print("="*70)
    print(f"階段 2 完成！")
    print(f"最終準確率: {final_acc*100:.2f}% ({correct}/{total})")
    print(f"完整結果已存至: {results_file}")
    print("="*70)
    with open(full_log, "a", encoding="utf-8") as fl:
        fl.write("\n" + "="*70 + "\n")
        fl.write(f"Stage 2 done. Accuracy: {final_acc*100:.2f}% ({correct}/{total})\n")
        fl.write(f"Results: {results_file}\n")
        fl.write("="*70 + "\n")
    
    # 清理記憶體
    del gen_model, gen_tok
    torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Instance-level FIPO for GSM8K - 兩階段執行版本（修正版）"
    )
    parser.add_argument("--optimizer_model", type=str, default="allenai/tulu-2-dpo-13b")
    parser.add_argument("--generator_model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--optimizer_4bit", action="store_true", help="Use 4-bit quantization for optimizer")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--out_prefix", type=str, default="gsm8k_instance", help="Output file prefix")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to store logs and outputs")
    parser.add_argument("--mode", type=str, choices=["optimize", "generate"], required=True,
                        help="optimize: 階段1(生成優化prompt) | generate: 階段2(用優化prompt生成答案)")
    parser.add_argument("--optimized_prompts_file", type=str, default=None,
                        help="For generate mode: path to optimized prompts JSONL file from stage 1")
    parser.add_argument("--generator_8bit", action="store_true", help="Load generator in 8-bit quantization")
    parser.add_argument("--generator_4bit", action="store_true", help="Load generator in 4-bit quantization")
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print("GPU cache reset\n")
    
    if args.mode == "optimize":
        optimize_prompts_only(
            optimizer_model_id=args.optimizer_model,
            use_4bit=args.optimizer_4bit,
            limit=args.limit,
            out_prefix=args.out_prefix,
            log_dir=args.log_dir,
        )
    elif args.mode == "generate":
        if args.optimized_prompts_file is None:
            raise ValueError("--optimized_prompts_file is required for generate mode")
        generate_with_optimized_prompts(
            generator_model_id=args.generator_model,
            optimized_prompts_file=args.optimized_prompts_file,
            limit=args.limit,
            out_prefix=args.out_prefix,
            log_dir=args.log_dir,
            gen_8bit=args.generator_8bit,
            gen_4bit=args.generator_4bit,
        )


if __name__ == "__main__":
    main()
