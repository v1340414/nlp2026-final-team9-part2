#!/usr/bin/env python3
"""P3 CMU/Ridge 실험의 generated sonnet 결과를 평가한다."""

import argparse
import csv
import json
import re
from pathlib import Path

import pronouncing
from sacrebleu.metrics import CHRF


RHYME_PAIRS = (
    (0, 2), (1, 3),
    (4, 6), (5, 7),
    (8, 10), (9, 11),
    (12, 13),
)


def load_gold_sonnets(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sonnets = re.split(r"\n\s*\d+\s*\n", text)[1:]
    return [sonnet.strip() for sonnet in sonnets]


def parse_generated_sonnets(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sonnets = []
    current_id = None
    current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("#") or line.startswith("--Generated"):
            continue
        if re.fullmatch(r"\s*\d+\s*", line):
            if current_id is not None:
                sonnets.append("\n".join(current_lines).strip())
            current_id = int(line.strip())
            current_lines = []
        elif current_id is not None:
            current_lines.append(line)

    if current_id is not None:
        sonnets.append("\n".join(current_lines).strip())
    return [sonnet for sonnet in sonnets if sonnet.strip()]


def parse_metadata(path):
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# {"):
            try:
                return json.loads(line[2:])
            except json.JSONDecodeError:
                return {}
    return {}


def nonempty_lines(text, max_lines=14):
    return [line.strip() for line in text.splitlines() if line.strip()][:max_lines]


def word_tokens(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def get_last_word(line):
    words = word_tokens(line)
    return words[-1] if words else None


def rhyme_part(word):
    if not word:
        return None
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    return pronouncing.rhyming_part(phones[0])


def do_rhyme(left, right):
    if left and right and left.lower() == right.lower():
        return False
    left_part = rhyme_part(left)
    right_part = rhyme_part(right)
    return bool(left_part and right_part and left_part == right_part)


def rhyme_acc(sonnet):
    lines = nonempty_lines(sonnet)
    if len(lines) < 14:
        return 0.0
    endings = [get_last_word(line) for line in lines]
    correct = sum(
        1
        for left, right in RHYME_PAIRS
        if endings[left] and endings[right] and do_rhyme(endings[left], endings[right])
    )
    return correct / len(RHYME_PAIRS)


def repetition_ratio(sonnet, n=3):
    words = word_tokens(" ".join(nonempty_lines(sonnet)))
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[idx:idx + n]) for idx in range(len(words) - n + 1)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams) if ngrams else 0.0


def compute_chrf(generated, gold):
    max_len = min(len(generated), len(gold))
    if max_len == 0:
        return None
    return float(CHRF().corpus_score(generated[:max_len], [gold[:max_len]]).score)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def gen_time(exp_dir, split):
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            value = metrics.get(f"{split}_generation_sec")
            if value is not None:
                return float(value)
        except json.JSONDecodeError:
            pass

    log_path = exp_dir / "generation.log"
    if not log_path.exists():
        return None
    total = 0.0
    count = 0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{split} sonnet ") and "elapsed_sec=" in line:
            match = re.search(r"elapsed_sec=([0-9.]+)", line)
            if match:
                total += float(match.group(1))
                count += 1
    return total if count else None


def evaluate_dir(exp_dir, split, gold):
    generated_path = exp_dir / f"generated_{split}.txt"
    generated = parse_generated_sonnets(generated_path)
    metadata = parse_metadata(generated_path)
    method = metadata.get("experiment") or exp_dir.name
    num_candidates = metadata.get("num_candidates")
    rerank_candidates = metadata.get("rerank_candidates")
    if str(method).startswith("prefix"):
        num_candidates = 0 if num_candidates is None else num_candidates
        rerank_candidates = 1 if rerank_candidates is None else rerank_candidates
    return {
        "Method": method,
        "temperature": metadata.get("temperature"),
        "top_p": metadata.get("top_p"),
        "num_candidates": num_candidates,
        "rerank_candidates": rerank_candidates,
        "chrF": compute_chrf(generated, gold),
        "Rhyme Acc": mean([rhyme_acc(text) for text in generated]),
        "Repetition": mean([repetition_ratio(text) for text in generated]),
        "Gen Time": gen_time(exp_dir, split),
    }


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--gold-dev", default="data/TRUE_sonnets_held_out_dev.txt")
    parser.add_argument("--gold-test", default="data/TRUE_sonnets_held_out.txt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gold_path = args.gold_dev if args.split == "dev" else args.gold_test
    gold = load_gold_sonnets(gold_path)
    predictions_dir = Path(args.predictions_dir)
    rows = []

    for exp_dir in sorted(path for path in predictions_dir.iterdir() if path.is_dir()):
        if (exp_dir / f"generated_{args.split}.txt").exists():
            rows.append(evaluate_dir(exp_dir, args.split, gold))

    columns = [
        "Method", "temperature", "top_p", "num_candidates", "rerank_candidates",
        "chrF", "Rhyme Acc", "Repetition", "Gen Time",
    ]
    output = Path(args.output or f"predictions/{args.split}_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column)) for column in columns})

    print(output)


if __name__ == "__main__":
    main()
