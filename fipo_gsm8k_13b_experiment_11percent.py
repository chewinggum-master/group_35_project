#!/usr/bin/env python3
"""
FIPO (13B optimizer) experiment for GSM8K with Llama2-7B as the generator.

What this script does:
- Option A: Generate a FIPO instruction once using a 13B local optimizer (Tulu-2-DPO-13B), optionally in 4-bit.
- Option B: Evaluate Llama2-7B on GSM8K test set using the saved FIPO instruction, with 3-shot prompting.

Notes:
- We place all logs and results under runs/ with the prefix 'fipo_gsm8k_13b'.
- We do NOT run a naive baseline here (per user request). Use your existing results for naive.
- 3-shot means: We prepend 3 fixed training examples to the prompt before each test question. These support examples are not part of the accuracy tally; we only score the test items.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)
import os, json, argparse, re, gc, random
from typing import List, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional 4-bit
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# CUDA/allocator safeguards (set before CUDA init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
torch.backends.cuda.matmul.allow_tf32 = True


def load_prompts_template() -> Dict[str, str]:
    """Load data/prompts.json to use the official FIPO optimizer template.
    Only the 'optimizer' field is used in this script.
    """
    path = os.path.join("data", "prompts.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_optimizer_input(naive_instruction: str, g_n_words: int = 60) -> str:
    """Build the meta prompt for the optimizer using the official template.

    We fill:
    - S_P with the naive instruction
    - O_C left empty (no optional silver/golden blocks for a general instruction)
    - G_N with a small word budget for concision
    """
    tmpl = load_prompts_template()["optimizer"]
    meta = (
        tmpl
        .replace("S_P", naive_instruction)
        .replace("O_C", "")
        .replace("G_N", str(g_n_words))
    )
    # Tulu chat format
    return f"<|user|>\n{meta}\n<|assistant|>\n"


@torch.inference_mode()
def generate_fipo_instruction(
    optimizer_model_id: str,
    naive_instruction: str,
    use_4bit: bool = True,
    max_input_len: int = 768,
    max_new_tokens: int = 256,
) -> str:
    print(f"\n[Step] Load Optimizer: {optimizer_model_id}", flush=True)
    torch.cuda.empty_cache(); gc.collect();

    tok = AutoTokenizer.from_pretrained(optimizer_model_id, use_fast=True)
    if use_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes not available; install it or disable --optimizer_4bit")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            quantization_config=bnb,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            optimizer_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()

    prompt = build_optimizer_input(naive_instruction)
    print(f"Optimizer input (head): {prompt[:160]}...", flush=True)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_len).to(model.device)
    print("Generating optimized instruction...", flush=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in text:
        fipo = text.split("<|assistant|>")[-1].strip()
    else:
        fipo = text.strip()

    del model, tok, inputs, outputs
    torch.cuda.empty_cache(); gc.collect()
    print("\n[OK] Optimized instruction generated.\n", flush=True)
    return fipo


def get_few_shot_examples(n: int = 3) -> List[Dict[str, str]]:
    if n <= 0:
        return []
    # Use full chain-of-thought answers, not just final numbers
    examples = [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fiber\n#### 3"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
        },
    ]
    return examples[:n]


def build_llama2_prompt_from_examples(instruction: str, question: str, examples: List[Dict[str, str]]) -> str:
    # Let FIPO instruction guide the format; examples show full reasoning
    parts = [instruction.strip(), ""]
    for ex in examples:
        parts.append(f"Question: {ex['question']}\n{ex['answer']}\n")
    parts.append(f"Question: {question}\n")
    return "\n".join(parts)


def extract_number(s: str) -> str | None:
    if not s:
        return None
    try:
        lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
        for ln in reversed(lines[-5:]):
            if re.search(r"(?i)(final\s*answer|^answer\s*:)", ln):
                nums = re.findall(r"-?\d+(?:\.\d+)?", ln)
                if nums:
                    return nums[-1]
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return m[-1] if m else None
    except Exception:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
        return m[-1] if m else None


def load_few_shot_examples(
    source: str = "static",
    k: int = 3,
    seed: int = 42,
    indices: Optional[List[int]] = None,
) -> List[Dict[str, str]]:
    """Load k few-shot exemplars with correct answers.

    - static: use the built-in verified examples.
    - gsm8k-train: sample k items from GSM8K train split and use their gold answers.
    - gsm8k-train (fixed indices): if indices provided, pick exactly those.
    """
    if k <= 0:
        return []
    if source == "static":
        return get_few_shot_examples(k)

    if source == "gsm8k-train":
        ds = load_dataset("gsm8k", "main", split="train")
        n = len(ds)
        picked: List[Dict[str, str]] = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                if idx < 0 or idx >= n:
                    continue
                item = ds[int(idx)]
                q, a = item["question"], item["answer"]
                # Keep full reasoning as example (answer field contains step-by-step solution)
                picked.append({"question": q, "answer": a})
            return picked[:k] if k is not None else picked
        else:
            k = min(k, n)
            rng = random.Random(seed)
            idxs = list(range(n))
            rng.shuffle(idxs)
            for idx in idxs:
                if len(picked) >= k:
                    break
                item = ds[idx]
                q, a = item["question"], item["answer"]
                # Keep full reasoning as example
                picked.append({"question": q, "answer": a})
            return picked

    raise ValueError(f"Unknown few-shot source: {source}")


@torch.inference_mode()
def evaluate_gsm8k(
    instruction: str,
    model_id: str = "meta-llama/Llama-2-7b-hf",
    num_shots: int = 3,
    limit: int | None = None,
    out_prefix: str = "fipo_gsm8k_13b",
    few_shot_source: str = "gsm8k-train",
    few_shot_seed: int = 42,
    few_shot_indices: Optional[str] = None,
):
    print("\n" + "=" * 70, flush=True)
    print(f"Evaluate: {model_id} | shots={num_shots}", flush=True)
    print(f"Instruction head: {instruction[:100]}...", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Clean up GPU
    torch.cuda.empty_cache(); gc.collect();

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()

    ds = load_dataset("gsm8k", "main", split="test")
    if limit is not None:
        ds = ds.select(range(limit))
    total = len(ds)
    print(f"Dataset size: {total}", flush=True)

    # Prepare few-shot exemplars once (deterministic if seed or indices fixed)
    indices_list: Optional[List[int]] = None
    if few_shot_indices:
        try:
            indices_list = [int(x) for x in few_shot_indices.split(",") if x.strip() != ""]
        except Exception:
            indices_list = None
    # If indices are provided, use their count as k to avoid mismatch.
    k = len(indices_list) if indices_list else num_shots
    shot_examples = load_few_shot_examples(
        source=few_shot_source,
        k=k,
        seed=few_shot_seed,
        indices=indices_list,
    )
    print(f"Few-shot source: {few_shot_source}, k={len(shot_examples)}, indices={indices_list}", flush=True)

    os.makedirs("runs", exist_ok=True)
    progress_jsonl = os.path.join("runs", f"{out_prefix}_progress_{total}.jsonl")
    progress_log = os.path.join("runs", f"{out_prefix}_progress_{total}.log")
    result_json = os.path.join("runs", f"{out_prefix}_{num_shots}shot_{total}.json")

    print(f"Progress JSONL: {progress_jsonl}", flush=True)
    print(f"Progress LOG  : {progress_log}", flush=True)
    open(progress_jsonl, "a", encoding="utf-8").close()
    with open(progress_log, "a", encoding="utf-8") as pf:
        pf.write(f"start total={total}\n")

    correct = 0
    preds = []

    for i, item in enumerate(ds):
        q, ans = item["question"], item["answer"]
        gold = extract_number(ans)
        prompt = build_llama2_prompt_from_examples(instruction, q, shot_examples)

        if i > 0 and i % 50 == 0:
            torch.cuda.empty_cache()

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        try:
            # Use fp16 autocast; greedy decoding for stability
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            # Retry with smaller lengths once
            torch.cuda.empty_cache(); gc.collect()
            try:
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                gen = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            except Exception as e2:
                gen = f"<GEN_ERR: {type(e2).__name__}: {str(e2)[:120]}>"

        # Trim at next question sentinel if present
        for stop_str in ["\n\nQuestion:", "\nQuestion:", "Question:"]:
            if stop_str in gen:
                gen = gen.split(stop_str)[0].strip()
                break

        pred = extract_number(gen)
        ok = int(pred == gold)
        correct += ok
        rec = {"idx": i, "question": q, "generated": gen, "pred_num": pred, "gold": gold, "correct": ok}
        preds.append(rec)

        try:
            with open(progress_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps({"idx": i, "pred_num": pred, "gold": gold, "correct": ok}, ensure_ascii=False) + "\n")
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            acc_now = correct / (i + 1)
            print(f"完成 {i+1}/{total}, 正確率: {acc_now*100:.2f}%", flush=True)
            try:
                with open(progress_log, "a", encoding="utf-8") as pf:
                    pf.write(f"step {i+1}/{total} acc={acc_now:.4f}\n")
            except Exception:
                pass

    acc = correct / total if total else 0.0
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_id,
            "mode": "fipo-13b",
            "num_shots": num_shots,
            "instruction": instruction,
            "few_shot_source": few_shot_source,
            "few_shot_seed": few_shot_seed,
            "few_shot_indices": indices_list,
            "few_shot_examples": shot_examples,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "predictions": preds,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"Final: acc={acc*100:.2f}% ({correct}/{total})", flush=True)
    print(f"Saved: {result_json}", flush=True)
    print("=" * 70 + "\n", flush=True)

    del model, tok, inputs
    torch.cuda.empty_cache(); gc.collect()
    return acc, result_json


def main():
    parser = argparse.ArgumentParser()
    # Optimizer generation
    parser.add_argument("--gen_instruction_only", action="store_true", help="Only generate and save FIPO instruction with 13B, then exit")
    parser.add_argument("--optimizer_model", type=str, default="allenai/tulu-2-dpo-13b")
    parser.add_argument("--optimizer_4bit", action="store_true", help="Load optimizer in 4-bit (recommended for 13B)")
    parser.add_argument("--optimizer_max_input_len", type=int, default=768)
    parser.add_argument("--optimizer_max_new_tokens", type=int, default=256)
    parser.add_argument("--save_fipo_path", type=str, default="data/fipo_instruction_13b.txt")

    # Evaluation
    parser.add_argument("--limit", type=int, default=None, help="How many GSM8K test items to evaluate; omit for full (1319)")
    parser.add_argument("--num_shots", type=int, default=3)
    parser.add_argument("--few_shot_source", type=str, default="gsm8k-train", choices=["static", "gsm8k-train"], help="Where to draw few-shot exemplars from")
    parser.add_argument("--few_shot_seed", type=int, default=42, help="Seed for sampling train exemplars when using gsm8k-train")
    parser.add_argument("--few_shot_indices", type=str, default=None, help="Comma-separated train indices to use as fixed few-shot exemplars (overrides sampling)")

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect();
        torch.cuda.synchronize()
        print("GPU cache reset", flush=True)

    # Base naive instruction to be optimized
    naive_instruction = "Solve the math word problem step by step, then output the final numeric answer."

    if args.gen_instruction_only:
        print("\nOnly-optimizer mode: generate and save FIPO instruction, then exit\n", flush=True)
        fipo = generate_fipo_instruction(
            optimizer_model_id=args.optimizer_model,
            naive_instruction=naive_instruction,
            use_4bit=args.optimizer_4bit,
            max_input_len=args.optimizer_max_input_len,
            max_new_tokens=args.optimizer_max_new_tokens,
        )
        os.makedirs(os.path.dirname(args.save_fipo_path) or ".", exist_ok=True)
        with open(args.save_fipo_path, "w", encoding="utf-8") as f:
            f.write(fipo)
        print(f"Saved FIPO instruction to {args.save_fipo_path}", flush=True)
        return

    # Evaluation path: read instruction (must exist or we fallback)
    if os.path.exists(args.save_fipo_path):
        with open(args.save_fipo_path, "r", encoding="utf-8") as f:
            fipo_inst = f.read().strip()
        print(f"Loaded FIPO instruction from {args.save_fipo_path}", flush=True)
    else:
        print("Warning: FIPO instruction file not found; generating on-the-fly with 13B (4-bit if requested).", flush=True)
        fipo_inst = generate_fipo_instruction(
            optimizer_model_id=args.optimizer_model,
            naive_instruction=naive_instruction,
            use_4bit=args.optimizer_4bit,
            max_input_len=args.optimizer_max_input_len,
            max_new_tokens=args.optimizer_max_new_tokens,
        )
        os.makedirs(os.path.dirname(args.save_fipo_path) or ".", exist_ok=True)
        with open(args.save_fipo_path, "w", encoding="utf-8") as f:
            f.write(fipo_inst)
        print(f"Saved FIPO instruction to {args.save_fipo_path}", flush=True)

    # Evaluate with Llama2-7B
    _acc, _path = evaluate_gsm8k(
        instruction=fipo_inst,
        model_id="meta-llama/Llama-2-7b-hf",
        num_shots=args.num_shots,
        limit=args.limit,
        out_prefix="fipo_gsm8k_13b",
        few_shot_source=args.few_shot_source,
        few_shot_seed=args.few_shot_seed,
        few_shot_indices=args.few_shot_indices,
    )


if __name__ == "__main__":
    main()
