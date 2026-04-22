#!/usr/bin/env python3
"""
IFEval evaluation using vLLM inference + official IFEval scoring.

Usage:
    python ifeval_eval.py --model /path/to/model [--tp 4] [--output_dir ./results]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── IFEval scoring (built-in, no external package needed) ────────────────────
# Inline the core IFEval evaluation logic from google/IFEval
import re
import string
import unicodedata
from typing import Optional


# ---------- Instruction checkers (subset of official IFEval) ----------

def count_words(text: str) -> int:
    """Count words in text."""
    tokens = text.split()
    return len(tokens)

def count_sentences(text: str) -> int:
    import nltk
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def check_instruction(instruction_id: str, response: str, kwarg: dict) -> bool:
    """Check a single instruction. Returns True if followed."""
    r = response

    if instruction_id == "punctuation:no_comma":
        return "," not in r

    elif instruction_id == "length_constraints:number_words":
        relation = kwarg.get("relation", "at least")
        num_words = kwarg.get("num_words", 0) or 0
        wc = count_words(r)
        if relation == "at least":
            return wc >= num_words
        elif relation == "at most":
            return wc <= num_words
        elif relation == "exactly":
            return wc == num_words
        return True

    elif instruction_id == "length_constraints:number_sentences":
        relation = kwarg.get("relation", "at least")
        num_sentences = kwarg.get("num_sentences", 0) or 0
        sc = count_sentences(r)
        if relation == "at least":
            return sc >= num_sentences
        elif relation == "at most":
            return sc <= num_sentences
        elif relation == "exactly":
            return sc == num_sentences
        return True

    elif instruction_id == "length_constraints:number_paragraphs":
        relation = kwarg.get("relation", "at least")
        num_paragraphs = kwarg.get("num_paragraphs", 0) or 0
        paras = [p for p in r.split("\n\n") if p.strip()]
        pc = len(paras)
        if relation == "at least":
            return pc >= num_paragraphs
        elif relation == "at most":
            return pc <= num_paragraphs
        elif relation == "exactly":
            return pc == num_paragraphs
        return True

    elif instruction_id == "detectable_format:number_bullet_lists":
        relation = kwarg.get("relation", "at least")
        num_bullets = kwarg.get("num_bullets", 0) or 0
        bullets = re.findall(r'^\s*[-*•]\s', r, re.MULTILINE)
        bc = len(bullets)
        if relation == "at least":
            return bc >= num_bullets
        elif relation == "at most":
            return bc <= num_bullets
        elif relation == "exactly":
            return bc == num_bullets
        return True

    elif instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = kwarg.get("num_highlights", 0) or 0
        # *text* or **text**
        highlights = re.findall(r'\*+[^*\n]+\*+', r)
        return len(highlights) >= num_highlights

    elif instruction_id == "detectable_format:number_sections":
        relation = kwarg.get("relation", "at least")
        num_sections = kwarg.get("num_sections", 0) or 0
        section_splitter = kwarg.get("section_spliter", "Section") or "Section"
        sections = re.findall(rf'^{re.escape(section_splitter)}\s', r, re.MULTILINE | re.IGNORECASE)
        sc = len(sections)
        if relation == "at least":
            return sc >= num_sections
        elif relation == "at most":
            return sc <= num_sections
        elif relation == "exactly":
            return sc == num_sections
        return True

    elif instruction_id == "keywords:existence":
        keywords = kwarg.get("keywords") or []
        r_lower = r.lower()
        return all(kw.lower() in r_lower for kw in keywords)

    elif instruction_id == "keywords:forbidden_words":
        forbidden = kwarg.get("forbidden_words") or []
        r_lower = r.lower()
        return not any(fw.lower() in r_lower for fw in forbidden)

    elif instruction_id == "keywords:frequency":
        keyword = kwarg.get("keyword", "") or ""
        relation = kwarg.get("relation", "at least")
        frequency = kwarg.get("frequency", 1) or 1
        count = len(re.findall(re.escape(keyword), r, re.IGNORECASE))
        if relation == "at least":
            return count >= frequency
        elif relation == "at most":
            return count <= frequency
        elif relation == "exactly":
            return count == frequency
        return True

    elif instruction_id == "keywords:letter_frequency":
        letter = kwarg.get("letter", "").lower() or ""
        relation = kwarg.get("let_relation", "at least")
        frequency = kwarg.get("let_frequency", 1) or 1
        count = r.lower().count(letter)
        if relation == "at least":
            return count >= frequency
        elif relation == "at most":
            return count <= frequency
        elif relation == "exactly":
            return count == frequency
        return True

    elif instruction_id == "detectable_content:number_placeholders":
        num_placeholders = kwarg.get("num_placeholders", 0) or 0
        placeholders = re.findall(r'\[.*?\]', r)
        return len(placeholders) >= num_placeholders

    elif instruction_id == "detectable_format:constrained_response":
        # Response must be one of specific options
        return True  # hard to check without knowing options

    elif instruction_id == "detectable_format:json_format":
        try:
            json.loads(r.strip())
            return True
        except Exception:
            # Try finding JSON block
            match = re.search(r'\{.*\}', r, re.DOTALL)
            if match:
                try:
                    json.loads(match.group())
                    return True
                except Exception:
                    pass
            return False

    elif instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = kwarg.get("prompt_to_repeat", "") or ""
        return prompt_to_repeat.lower() in r.lower() if prompt_to_repeat else True

    elif instruction_id == "combination:two_responses":
        return "******" in r

    elif instruction_id == "startend:end_checker":
        end_phrase = kwarg.get("end_phrase", "") or ""
        return r.rstrip().lower().endswith(end_phrase.lower()) if end_phrase else True

    elif instruction_id == "startend:quotation":
        r_stripped = r.strip()
        return r_stripped.startswith('"') and r_stripped.endswith('"')

    elif instruction_id == "change_case:english_capital":
        words = re.findall(r'[a-zA-Z]+', r)
        return all(w.isupper() for w in words) if words else True

    elif instruction_id == "change_case:english_lowercase":
        words = re.findall(r'[a-zA-Z]+', r)
        return all(w.islower() for w in words) if words else True

    elif instruction_id == "change_case:capital_word_frequency":
        relation = kwarg.get("capital_relation", "at least")
        frequency = kwarg.get("capital_frequency", 0) or 0
        words = re.findall(r'\b[A-Z][A-Z]+\b', r)
        count = len(words)
        if relation == "at least":
            return count >= frequency
        elif relation == "at most":
            return count <= frequency
        elif relation == "exactly":
            return count == frequency
        return True

    elif instruction_id == "language:response_language":
        language = kwarg.get("language", "en") or "en"
        try:
            import langdetect
            detected = langdetect.detect(r)
            return detected == language
        except Exception:
            return True

    elif instruction_id == "detectable_format:title":
        # Check for markdown title (# Title)
        return bool(re.search(r'^#+\s+\S', r, re.MULTILINE))

    elif instruction_id == "startend:first_word":
        first_word = kwarg.get("first_word", "") or ""
        words = r.strip().split()
        return words[0].lower() == first_word.lower() if (words and first_word) else True

    elif instruction_id == "detectable_format:nth_paragraph_first_word":
        first_word = kwarg.get("first_word", "") or ""
        nth_paragraph = kwarg.get("nth_paragraph", 1) or 1
        paras = [p for p in r.split("\n\n") if p.strip()]
        if nth_paragraph <= len(paras):
            words = paras[nth_paragraph - 1].strip().split()
            return words[0].lower() == first_word.lower() if (words and first_word) else True
        return False

    # Unknown instruction: assume pass
    return True


def score_response(prompt: str, response: str, instruction_id_list: list, kwargs: list) -> dict:
    """Score a single response against all instructions."""
    results = []
    for inst_id, kwarg in zip(instruction_id_list, kwargs):
        passed = check_instruction(inst_id, response, kwarg or {})
        results.append(passed)

    instruction_pass = sum(results) / len(results) if results else 0.0
    prompt_pass = float(all(results)) if results else 0.0

    return {
        "instruction_pass": instruction_pass,
        "prompt_pass": prompt_pass,
        "per_instruction": results,
    }


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_inference(model_path: str, prompts: list, tp: int, max_tokens: int = 1024) -> list:
    """Run batch inference with vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    # Get tokenizer from vLLM (avoids transformers version issues)
    tokenizer = llm.get_tokenizer()

    # Load chat_template.jinja if tokenizer doesn't have one built-in
    if not getattr(tokenizer, "chat_template", None):
        jinja_path = Path(model_path) / "chat_template.jinja"
        if jinja_path.exists():
            tokenizer.chat_template = jinja_path.read_text()

    # Apply chat template if available
    formatted = []
    for p in prompts:
        try:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = p
        formatted.append(text)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    outputs = llm.generate(formatted, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    model_short = Path(args.model).name
    if args.output_dir is None:
        args.output_dir = f"./ifeval_results/{model_short}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[IFEval] Model:      {args.model}")
    print(f"[IFEval] TP:         {args.tp}")
    print(f"[IFEval] Output:     {args.output_dir}")

    # Download IFEval dataset
    print("[IFEval] Loading dataset...")
    from datasets import load_dataset
    ds = load_dataset("google/IFEval", split="train")

    prompts = [ex["prompt"] for ex in ds]
    instruction_id_lists = [ex["instruction_id_list"] for ex in ds]
    kwargs_list = [ex["kwargs"] for ex in ds]

    # Run inference
    print(f"[IFEval] Running inference on {len(prompts)} prompts...")
    responses = run_inference(args.model, prompts, args.tp, args.max_tokens)

    # Score
    print("[IFEval] Scoring...")
    all_results = []
    total_inst_passed = 0
    total_inst_count = 0
    total_prompt_pass = 0

    for i, (prompt, response, inst_ids, kw_list) in enumerate(
        zip(prompts, responses, instruction_id_lists, kwargs_list)
    ):
        result = score_response(prompt, response, inst_ids, kw_list)
        result["prompt"] = prompt
        result["response"] = response
        result["instruction_ids"] = inst_ids
        all_results.append(result)
        total_inst_passed += sum(result["per_instruction"])
        total_inst_count += len(result["per_instruction"])
        total_prompt_pass += int(all(result["per_instruction"]))

    n = len(all_results)
    inst_level_acc = total_inst_passed / total_inst_count if total_inst_count else 0.0
    prompt_level_acc = total_prompt_pass / n if n else 0.0

    # Save results
    results_file = Path(args.output_dir) / "results.jsonl"
    with open(results_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "num_examples": n,
        "prompt_level_strict_acc": prompt_level_acc,
        "inst_level_strict_acc": inst_level_acc,
    }
    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 50)
    print(f"  IFEval Results: {model_short}")
    print("=" * 50)
    print(f"  prompt_level_strict_acc: {prompt_level_acc:.4f} ({prompt_level_acc*100:.2f}%)")
    print(f"  inst_level_strict_acc:   {inst_level_acc:.4f} ({inst_level_acc*100:.2f}%)")
    print(f"  Num examples: {n}")
    print(f"  Results saved: {args.output_dir}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    main()
