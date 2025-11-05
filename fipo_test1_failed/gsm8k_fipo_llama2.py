#!/usr/bin/env python3
"""
正確的 FIPO 實驗設計:
1. 用 Llama2-7B + naive prompt (3-shot) → baseline 分數
2. 用 Tulu-13B optimizer 優化 prompt
3. 用 Llama2-7B + FIPO prompt (3-shot) → 改進的分數

目標: 證明 FIPO 優化的 prompt 可以讓**同一個模型**產生更好的答案
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import re, json, os, argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # 延後到用到時再檢查
import gc

torch.backends.cuda.matmul.allow_tf32 = True

# 調整 CUDA 配置，降低碎片化與改善錯誤定位（在任何 CUDA 呼叫前設定）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

def extract_number(s):
    """更健壯的數字抽取：
    1) 優先從包含 'Final Answer' 或 'Answer:' 的最後幾行抓取數字
    2) 否則退回到全文的最後一個數字
    """
    if not s:
        return None
    try:
        lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
        # 從最後 5 行往回找帶有答案關鍵字的行
        for ln in reversed(lines[-5:]):
            if re.search(r"(?i)(final\s*answer|^answer\s*:)", ln):
                nums = re.findall(r"-?\d+(?:\.\d+)?", ln)
                if nums:
                    return nums[-1]
        # 沒找到就回退到全文最後一個數字
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return m[-1] if m else None
    except Exception:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return m[-1] if m else None

def get_few_shot_examples(num_shots=3):
    if num_shots == 0:
        return []
    examples = [
        {"question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "answer": "18"},
        {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": "3"},
        {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": "70000"},
    ]
    return examples[:num_shots]

def build_llama2_prompt(instruction, question, num_shots=3):
    """
    Llama2 格式的 prompt (不使用 chat template)
    """
    examples = get_few_shot_examples(num_shots)
    # 明確要求最終以 Answer: <number> 結尾，避免解析抓到過程中的數字
    output_rule = "Output only the final numeric answer on the last line as 'Answer: <number>' with no extra text after it."
    prompt = instruction.strip() + "\n\n" + output_rule + "\n\n"
    for ex in examples:
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    return prompt

@torch.inference_mode()
def generate_fipo_instruction(
    optimizer_model_id,
    naive_instruction,
    use_4bit: bool = False,
    opt_max_new_tokens: int = 256,
    opt_max_input_len: int = 512,
):
    """使用 Tulu-13B optimizer 生成優化後的指令"""
    print(f"\n正在載入 Optimizer: {optimizer_model_id}...", flush=True)
    
    # 強制清理 GPU 記憶體
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    tok = AutoTokenizer.from_pretrained(optimizer_model_id, use_fast=True)
    # 優先嘗試 4-bit 量化以避免 13B OOM
    if use_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("需要 transformers 的 BitsAndBytesConfig；請先安裝 bitsandbytes: pip install bitsandbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    model.eval()
    
    meta_prompt = "Rewrite the following instruction to make it more specific and effective for solving math word problems:"
    user_content = f"{meta_prompt}\n\nOriginal instruction: {naive_instruction}\n\nImproved instruction:"
    prompt = f"<|user|>\n{user_content}\n<|assistant|>\n"
    
    print(f"Optimizer prompt: {prompt[:150]}...", flush=True)
    
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=opt_max_input_len).to(model.device)
    print("生成優化指令...", flush=True)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=opt_max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in gen:
        fipo_inst = gen.split("<|assistant|>")[-1].strip()
    else:
        fipo_inst = gen.strip()
    
    del model, tok, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n生成的 FIPO 指令:\n{fipo_inst}\n", flush=True)
    return fipo_inst

@torch.inference_mode()
def evaluate_gsm8k_llama2(model_id, instruction, mode="naive", num_shots=3, limit=None, out_filename=None):
    """用 Llama2-7B 評測 GSM8K"""
    print(f"\n{'='*70}", flush=True)
    print(f"評測: {model_id} | 模式: {mode} | Shots: {num_shots}", flush=True)
    print(f"指令: {instruction[:80]}...", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    # 強制清理 GPU 記憶體
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    
    print(f"載入 Llama2-7B 模型...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    # 使用 fp16 並強制使用 eager attention，避免某些 GPU 上的閃存/自訂 kernel 造成非法存取
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    
    print("載入 GSM8K...", flush=True)
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(limit))
        print(f"限制為 {limit} 題", flush=True)
    
    total_len = len(ds)
    print(f"開始評測 {total_len} 題...\n", flush=True)
    
    # 準備進度輸出檔案（便於 tail 追蹤）
    os.makedirs("runs", exist_ok=True)
    progress_jsonl = f"runs/gsm8k_llama2-7b_{mode}_{num_shots}shot_progress_{total_len}.jsonl"
    progress_txt = f"runs/gsm8k_{mode}_{num_shots}shot_progress_{total_len}.log"
    print(f"進度檔: JSONL -> {progress_jsonl}", flush=True)
    print(f"進度檔: LOG   -> {progress_txt}", flush=True)
    # 開始前建立檔案並寫入簡短標頭，方便外部 tail 立即看到
    os.makedirs("runs", exist_ok=True)
    # 先建立空的 JSONL 檔（之後每題追加一行）
    try:
        open(progress_jsonl, "a", encoding="utf-8").close()
    except Exception:
        pass
    with open(progress_txt, "a", encoding="utf-8") as pf:
        pf.write(f"start total={total_len}\n")
    correct, preds = 0, []
    
    for i, item in enumerate(ds):
        q, ans = item["question"], item["answer"]
        gold = extract_number(ans)
        prompt = build_llama2_prompt(instruction, q, num_shots)
        
        # 每 50 題清理一次避免碎片累積
        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()
        
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        # 生成步驟加上 autocast，並在失敗時進行一次降規重試
        try:
            # 新 API：torch.amp.autocast(device_type='cuda', dtype=...)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
        except Exception as e:
            # 清理後用較小的輸入/輸出重試一次，避免偶發的非法存取
            torch.cuda.empty_cache()
            gc.collect()
            try:
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
            except Exception as e2:
                gen_text = f"<GENERATION_ERROR: {type(e2).__name__}: {str(e2)[:120]}>"
                pred = extract_number(gen_text)
                ok = int(pred is not None and pred == gold)
                correct += ok
                preds.append({
                    "idx": i,
                    "question": q,
                    "generated": gen_text,
                    "pred_num": pred,
                    "gold": gold,
                    "correct": ok,
                    "error": str(e2),
                })
                # 同步寫入進度 JSONL
                try:
                    with open(progress_jsonl, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps({"idx": i, "pred_num": pred, "gold": gold, "correct": ok, "error": str(e2)[:200]}, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                # 顯示本題失敗並繼續下一題
                print(f"[WARN] 第 {i+1} 題生成失敗，已跳過並記錄。", flush=True)
                continue
        gen_text = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Stop strings
        for stop_str in ["\n\nQuestion:", "\nQuestion:", "Question:"]:
            if stop_str in gen_text:
                gen_text = gen_text.split(stop_str)[0].strip()
                break
        
        pred = extract_number(gen_text)
        ok = int(pred == gold)
        correct += ok
        preds.append({"idx": i, "question": q, "generated": gen_text, "pred_num": pred, "gold": gold, "correct": ok})
        
        # 立刻追加一筆到 JSONL 以便外部追蹤
        try:
            with open(progress_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps({
                    "idx": i,
                    "pred_num": pred,
                    "gold": gold,
                    "correct": ok
                }, ensure_ascii=False) + "\n")
        except Exception as _:
            pass
        
        if (i + 1) % 10 == 0:
            acc_now = correct/(i+1)
            print(f"完成 {i+1}/{total_len}, 正確率: {acc_now*100:.2f}%", flush=True)
            # 同步到文字進度檔
            try:
                with open(progress_txt, "a", encoding="utf-8") as pf:
                    pf.write(f"step {i+1}/{total_len} acc={acc_now:.4f}\n")
            except Exception as _:
                pass
    
    acc = correct / total_len
    if out_filename:
        # 若指定自訂檔名，強制寫到 runs 底下
        out_path = os.path.join("runs", out_filename)
    else:
        out_path = f"runs/gsm8k_llama2-7b_{mode}_{num_shots}shot_{total_len}.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "mode": mode, "num_shots": num_shots, "instruction": instruction, "accuracy": acc, "correct": correct, "total": len(ds), "predictions": preds}, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}", flush=True)
    print(f"結果: {mode} | 準確率: {acc*100:.2f}% ({correct}/{len(ds)})", flush=True)
    print(f"檔案: {out_path}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    del model, tok, inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return acc, out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="限制評測題數")
    parser.add_argument("--num_shots", type=int, default=3, help="Few-shot 數量")
    parser.add_argument("--skip_naive", action="store_true", help="跳過 naive baseline")
    parser.add_argument("--skip_optimizer", action="store_true", help="跳過 optimizer，使用預設 FIPO 指令")
    parser.add_argument("--optimizer_model", type=str, default="allenai/tulu-2-dpo-13b", help="Optimizer model id (e.g. allenai/tulu-2-dpo-7b)")
    parser.add_argument("--optimizer_4bit", action="store_true", help="以 4-bit 量化載入 Optimizer（建議 13B 使用）")
    parser.add_argument("--optimizer_max_new_tokens", type=int, default=256, help="Optimizer 生成的最大新 token 數")
    parser.add_argument("--optimizer_max_input_len", type=int, default=512, help="Optimizer 輸入最大長度")
    parser.add_argument("--only_optimizer", action="store_true", help="只用優化器產生並保存 FIPO 指令，然後結束")
    parser.add_argument("--save_fipo_path", type=str, default="data/fipo_instruction.txt", help="保存/讀取 FIPO 指令的檔案路徑")
    parser.add_argument("--out_filename", type=str, default=None, help="自訂最終 JSON 檔名（會存到 runs/ 目錄下）")
    args = parser.parse_args()

    # 程式啟動時強制重置 GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print("GPU 記憶體已重置", flush=True)

    generator_model = "meta-llama/Llama-2-7b-hf"
    optimizer_model = args.optimizer_model
    
    print("="*70, flush=True)
    print("FIPO 實驗: Llama2-7B on GSM8K", flush=True)
    print(f"Generator: {generator_model}", flush=True)
    print(f"Optimizer: {optimizer_model}", flush=True)
    print(f"Limit: {args.limit}, Shots: {args.num_shots}", flush=True)
    print("="*70, flush=True)
    
    naive_instruction = "Solve the math word problem step by step, then output the final numeric answer."

    # 只跑優化器：直接生成並保存 FIPO 指令，然後結束（不跑任何 baseline 或評測）
    if args.only_optimizer:
        print("\n只執行優化器模式：不跑 baseline/評測，僅生成並保存 FIPO 指令\n", flush=True)
        try:
            fipo_instruction = generate_fipo_instruction(
                optimizer_model,
                naive_instruction,
                use_4bit=args.optimizer_4bit,
                opt_max_new_tokens=args.optimizer_max_new_tokens,
                opt_max_input_len=args.optimizer_max_input_len,
            )
            os.makedirs(os.path.dirname(args.save_fipo_path) or ".", exist_ok=True)
            with open(args.save_fipo_path, "w", encoding="utf-8") as f:
                f.write(fipo_instruction)
            print(f"已保存 FIPO 指令至 {args.save_fipo_path}", flush=True)
        except Exception as e:
            print(f"優化器失敗: {e}", flush=True)
            sys.exit(1)
        return
    
    # 步驟 1: Naive Baseline
    if not args.skip_naive:
        print("\n第 1 步: Naive Baseline (Llama2-7B)\n", flush=True)
        naive_acc, naive_path = evaluate_gsm8k_llama2(generator_model, naive_instruction, "naive", args.num_shots, args.limit)
        print(f"✓ Naive: {naive_acc:.2%}", flush=True)
    else:
        naive_acc = None
        print("跳過 Naive baseline", flush=True)
    
    # 步驟 2: 生成 FIPO 指令
    if not args.skip_optimizer:
        print("\n第 2 步: 用 Tulu Optimizer 生成 FIPO 指令\n", flush=True)
        try:
            fipo_instruction = generate_fipo_instruction(
                optimizer_model,
                naive_instruction,
                use_4bit=args.optimizer_4bit,
                opt_max_new_tokens=args.optimizer_max_new_tokens,
                opt_max_input_len=args.optimizer_max_input_len,
            )
            # 保存到檔案以便後續使用
            os.makedirs(os.path.dirname(args.save_fipo_path) or ".", exist_ok=True)
            with open(args.save_fipo_path, "w", encoding="utf-8") as f:
                f.write(fipo_instruction)
            print(f"已保存 FIPO 指令至 {args.save_fipo_path}", flush=True)
            if args.only_optimizer:
                print("只執行優化器模式完成，程式結束。", flush=True)
                return
        except Exception as e:
            print(f"警告: Optimizer 失敗: {e}", flush=True)
            print("使用預設 FIPO 指令...", flush=True)
            fipo_instruction = "Solve this mathematics problem by breaking it down into clear steps. For each step, explain your reasoning and calculations. Finally, provide the exact numerical answer."
    else:
        print("\n使用預設或已保存的 FIPO 指令\n", flush=True)
        # 嘗試從檔案讀取
        if os.path.exists(args.save_fipo_path):
            with open(args.save_fipo_path, "r", encoding="utf-8") as f:
                fipo_instruction = f.read().strip()
            print(f"已從 {args.save_fipo_path} 讀取", flush=True)
        else:
            fipo_instruction = "Solve this mathematics problem by breaking it down into clear steps. For each step, explain your reasoning and calculations. Finally, provide the exact numerical answer."
    
    print(f"✓ FIPO 指令: {fipo_instruction}", flush=True)
    
    # 步驟 3: FIPO 評測
    print("\n第 3 步: FIPO 評測 (Llama2-7B)\n", flush=True)
    fipo_acc, fipo_path = evaluate_gsm8k_llama2(
        generator_model,
        fipo_instruction,
        "fipo",
        args.num_shots,
        args.limit,
        out_filename=args.out_filename,
    )
    print(f"✓ FIPO: {fipo_acc:.2%}", flush=True)
    
    # 最終總結
    print("\n" + "="*70, flush=True)
    print("最終結果 (Llama2-7B on GSM8K)", flush=True)
    print("="*70, flush=True)
    if naive_acc is not None:
        print(f"Naive Prompt:    {naive_acc*100:.2f}%", flush=True)
        print(f"FIPO Prompt:     {fipo_acc*100:.2f}%", flush=True)
        print(f"改進幅度:        {(fipo_acc-naive_acc)*100:+.2f}% (絕對)", flush=True)
        if naive_acc > 0:
            print(f"                 {((fipo_acc-naive_acc)/naive_acc)*100:+.2f}% (相對)", flush=True)
        print(f"\n對比論文 (Llama2-7B GSM8K):", flush=True)
        print(f"  論文 Naive:    8.89%", flush=True)
        print(f"  論文 FIPO:     11.70%", flush=True)
        print(f"  論文改進:      +31.6% (相對)", flush=True)
    else:
        print(f"FIPO Prompt:     {fipo_acc*100:.2f}%", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
