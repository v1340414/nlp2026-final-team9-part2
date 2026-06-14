#!/usr/bin/env python3
"""P3 checkpoint 기반 prefix, CMU, CMU+Ridge decoding 실험을 실행한다."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEV_PROMPT_PATH = ROOT / "data" / "sonnets_held_out_dev.txt"
TEST_PROMPT_PATH = ROOT / "data" / "sonnets_held_out.txt"
DEV_GOLD_PATH = ROOT / "data" / "TRUE_sonnets_held_out_dev.txt"

sg = None
SonnetsDataset = None
generate_rhyming_sonnet = None
chrf_target = None
extract_features = None
features_to_vector = None
fit_ridge_reranker = None
load_ridge_model = None
rerank_sonnets = None
save_ridge_fit = None

EXPERIMENTS = {
  "prefix_standard": {
    "method": "prefix",
    "temperature": 0.8,
    "top_p": 0.90,
    "num_candidates": 0,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "cmu_balanced": {
    "method": "cmu",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 10,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "cmu_balanced_n20": {
    "method": "cmu",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 20,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "cmu_balanced_n30": {
    "method": "cmu",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 30,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "cmu_standard": {
    "method": "cmu",
    "temperature": 0.8,
    "top_p": 0.90,
    "num_candidates": 10,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "cmu_creative": {
    "method": "cmu",
    "temperature": 1.0,
    "top_p": 0.95,
    "num_candidates": 10,
    "rerank_candidates": 1,
    "ridge_train_candidates": 0,
  },
  "rerank_balanced": {
    "method": "cmu_ridge",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 10,
    "rerank_candidates": 3,
    "ridge_train_candidates": 3,
  },
  "rerank_balanced_n20": {
    "method": "cmu_ridge",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 20,
    "rerank_candidates": 3,
    "ridge_train_candidates": 3,
  },
  "rerank_balanced_n30": {
    "method": "cmu_ridge",
    "temperature": 0.7,
    "top_p": 0.85,
    "num_candidates": 30,
    "rerank_candidates": 3,
    "ridge_train_candidates": 3,
  },
  "rerank_standard": {
    "method": "cmu_ridge",
    "temperature": 0.8,
    "top_p": 0.90,
    "num_candidates": 10,
    "rerank_candidates": 3,
    "ridge_train_candidates": 3,
  },
}


def import_runtime_dependencies(needs_cmu=False, needs_ridge=False):
  global sg
  global SonnetsDataset
  global generate_rhyming_sonnet
  global chrf_target
  global extract_features
  global features_to_vector
  global fit_ridge_reranker
  global load_ridge_model
  global rerank_sonnets
  global save_ridge_fit

  if sg is not None and (not needs_cmu or generate_rhyming_sonnet is not None):
    return

  if sg is None:
    import sonnet_generation as sg_module
    from datasets import SonnetsDataset as SonnetsDatasetClass
    sg = sg_module
    SonnetsDataset = SonnetsDatasetClass

  if needs_cmu and generate_rhyming_sonnet is None:
    from rhyme_decoding import generate_rhyming_sonnet as generate_rhyming_sonnet_fn
    generate_rhyming_sonnet = generate_rhyming_sonnet_fn

  if needs_ridge and rerank_sonnets is None:
    from ridge_reranker import (
      chrf_target as chrf_target_fn,
      extract_features as extract_features_fn,
      features_to_vector as features_to_vector_fn,
      fit_ridge_reranker as fit_ridge_reranker_fn,
      load_ridge_model as load_ridge_model_fn,
      rerank_sonnets as rerank_sonnets_fn,
      save_ridge_fit as save_ridge_fit_fn,
    )
    chrf_target = chrf_target_fn
    extract_features = extract_features_fn
    features_to_vector = features_to_vector_fn
    fit_ridge_reranker = fit_ridge_reranker_fn
    load_ridge_model = load_ridge_model_fn
    rerank_sonnets = rerank_sonnets_fn
    save_ridge_fit = save_ridge_fit_fn


def write_log(message, log_path):
  with open(log_path, "a", encoding="utf-8") as f:
    f.write(message.rstrip() + "\n")


def add_metadata_header(output_path, metadata):
  path = Path(output_path)
  body = path.read_text(encoding="utf-8")
  header = [
    "# Experiment Metadata",
    "# " + json.dumps(metadata, sort_keys=True, ensure_ascii=False),
    "",
  ]
  path.write_text("\n".join(header) + body, encoding="utf-8")


def write_generated_sonnets(output_path, generated_sonnets):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as f:
    f.write("--Generated Sonnets-- \n\n")
    for sonnet_id, sonnet_text in generated_sonnets:
      f.write(f"\n{sonnet_id}\n")
      f.write(sonnet_text.strip())


def generate_prefix_split(model, exp, args, split_name, prompt_path, output_path, log_path):
  dataset = SonnetsDataset(str(prompt_path))
  generated = []
  start = time.time()

  for idx, (sonnet_id, prompt_text) in enumerate(dataset):
    item_start = time.time()
    write_log(f"{split_name} sonnet {idx + 1}/{len(dataset)}: prefix generation", log_path)
    encoding = model.tokenizer(
      prompt_text,
      return_tensors="pt",
      padding=True,
      truncation=True,
    ).to(model.get_device())
    _, continuation = model.generate(
      encoding["input_ids"],
      temperature=exp["temperature"],
      top_p=exp["top_p"],
      max_length=args.max_length,
      target_lines=args.target_lines,
    )
    sonnet_text = sg.keep_first_nonempty_lines(
      prompt_text + continuation,
      args.target_lines,
    )
    generated.append((sonnet_id, sonnet_text))
    write_log(
      f"{split_name} sonnet {sonnet_id}: elapsed_sec={time.time() - item_start:.2f}",
      log_path,
    )

  write_generated_sonnets(output_path, generated)
  return generated, time.time() - start


def generate_cmu_candidate(model, prompt, exp):
  return generate_rhyming_sonnet(
    model,
    prompt,
    num_candidates=exp["num_candidates"],
    temperature=exp["temperature"],
    top_p=exp["top_p"],
    max_line_tokens=22,
    min_line_tokens=4,
    soft_target_tokens=8,
    nl_boost=2.0,
    penalize_identical=False,
    verbose=False,
    rep_penalty=1.3,
    no_repeat_ngram=3,
  )["text"]


def train_or_load_ridge(model, exp, exp_dir, log_path):
  ridge_path = exp_dir / "ridge_model.json"
  if ridge_path.exists():
    write_log(f"Loaded existing Ridge model: {ridge_path}", log_path)
    return load_ridge_model(str(ridge_path)), str(ridge_path), False

  prompt_dataset = SonnetsDataset(str(DEV_PROMPT_PATH))
  gold_dataset = SonnetsDataset(str(DEV_GOLD_PATH))
  num_prompts = min(len(prompt_dataset), len(gold_dataset))
  feature_rows = []
  targets = []

  start = time.time()
  for idx in range(num_prompts):
    _, prompt_text = prompt_dataset[idx]
    _, gold_text = gold_dataset[idx]
    write_log(f"Ridge train prompt {idx + 1}/{num_prompts}", log_path)
    for _ in range(exp["ridge_train_candidates"]):
      candidate = generate_cmu_candidate(model, prompt_text, exp)
      features = extract_features(candidate, prompt=prompt_text)
      feature_rows.append(features_to_vector(features))
      targets.append(chrf_target(candidate, gold_text))

  ridge_fit = fit_ridge_reranker(feature_rows, targets, alpha=1.0)
  save_ridge_fit(str(ridge_path), ridge_fit)
  write_log(
    f"Trained Ridge: examples={ridge_fit.num_examples}, "
    f"target_mean={ridge_fit.target_mean:.4f}, train_mae={ridge_fit.train_mae:.4f}, "
    f"elapsed_sec={time.time() - start:.2f}",
    log_path,
  )
  coef_log = ", ".join(
    f"{name}={coef:.4f}"
    for name, coef in zip(ridge_fit.model.feature_names, ridge_fit.model.coef)
  )
  write_log(f"Ridge coefficients: {coef_log}", log_path)
  return ridge_fit.model, str(ridge_path), True


def generate_cmu_split(model, exp, split_name, prompt_path, output_path, log_path, ridge_model=None):
  dataset = SonnetsDataset(str(prompt_path))
  generated = []
  start = time.time()

  for idx, (sonnet_id, prompt_text) in enumerate(dataset):
    item_start = time.time()
    candidate_count = exp["rerank_candidates"] if ridge_model is not None else 1
    write_log(
      f"{split_name} sonnet {idx + 1}/{len(dataset)}: candidates={candidate_count}",
      log_path,
    )
    candidates = [
      generate_cmu_candidate(model, prompt_text, exp)
      for _ in range(candidate_count)
    ]

    if ridge_model is not None:
      reranked = rerank_sonnets(candidates, prompt=prompt_text, model=ridge_model)
      sonnet_text = reranked.text
      write_log(
        f"{split_name} sonnet {sonnet_id}: selected={reranked.candidate_index + 1}/"
        f"{candidate_count}, predicted_chrf={reranked.score:.4f}",
        log_path,
      )
    else:
      sonnet_text = candidates[0]

    generated.append((sonnet_id, sonnet_text))
    write_log(
      f"{split_name} sonnet {sonnet_id}: elapsed_sec={time.time() - item_start:.2f}",
      log_path,
    )

  write_generated_sonnets(output_path, generated)
  return generated, time.time() - start


def run_experiment(args):
  exp = EXPERIMENTS[args.experiment]
  exp_dir = Path(args.output_root) / args.experiment

  if args.dry_run:
    print(json.dumps({
      "experiment": args.experiment,
      "checkpoint": str(args.checkpoint),
      "output_dir": str(exp_dir),
      **exp,
    }, indent=2))
    return

  exp_dir.mkdir(parents=True, exist_ok=True)
  log_path = exp_dir / "generation.log"
  log_path.write_text("", encoding="utf-8")

  import_runtime_dependencies(
    needs_cmu=exp["method"] in ("cmu", "cmu_ridge"),
    needs_ridge=exp["method"] == "cmu_ridge",
  )

  sg.LOG_TO_CONSOLE = False
  sg.TQDM_DISABLE = True
  sg.seed_everything(args.seed)

  checkpoint = Path(args.checkpoint)
  if not checkpoint.exists():
    raise FileNotFoundError(
      f"Checkpoint를 찾을 수 없습니다: {checkpoint}. P3 selected checkpoint를 먼저 해당 위치에 두세요."
    )

  device = "cuda" if args.use_gpu else "cpu"
  write_log(f"Loading checkpoint: {checkpoint}", log_path)
  model = sg.load_sonnet_model_from_checkpoint(str(checkpoint), sg.torch.device(device))
  model.eval()

  metadata = {
    "experiment": args.experiment,
    "method": exp["method"],
    "checkpoint": str(checkpoint),
    "model_size": "gpt2-xl",
    "prefix_length": 20,
    "prefix_hidden_size": 512,
    "prefix_source": "P3_prefix20_hidden512_lr2e-4",
    "seed": args.seed,
    "temperature": exp["temperature"],
    "top_p": exp["top_p"],
    "max_length": args.max_length,
    "target_lines": args.target_lines,
    "num_candidates": exp["num_candidates"],
    "rerank_candidates": exp["rerank_candidates"],
    "ridge_train_candidates": exp["ridge_train_candidates"],
  }

  ridge_model = None
  if exp["method"] == "cmu_ridge":
    ridge_model, ridge_path, trained = train_or_load_ridge(model, exp, exp_dir, log_path)
    metadata["ridge_model_path"] = ridge_path
    metadata["ridge_trained_this_run"] = trained

  dev_generated = []
  test_generated = []
  dev_sec = None
  test_sec = None

  if exp["method"] == "prefix":
    if args.split in ("dev", "both"):
      dev_generated, dev_sec = generate_prefix_split(
        model, exp, args, "dev", DEV_PROMPT_PATH, exp_dir / "generated_dev.txt", log_path
      )
    if args.split in ("test", "both"):
      test_generated, test_sec = generate_prefix_split(
        model, exp, args, "test", TEST_PROMPT_PATH, exp_dir / "generated_test.txt", log_path
      )
  else:
    if args.split in ("dev", "both"):
      dev_generated, dev_sec = generate_cmu_split(
        model, exp, "dev", DEV_PROMPT_PATH, exp_dir / "generated_dev.txt", log_path, ridge_model
      )
    if args.split in ("test", "both"):
      test_generated, test_sec = generate_cmu_split(
        model, exp, "test", TEST_PROMPT_PATH, exp_dir / "generated_test.txt", log_path, ridge_model
      )

  metrics = {
    **metadata,
    "dev_generation_sec": dev_sec,
    "test_generation_sec": test_sec,
    "dev_num_sonnets": len(dev_generated) if dev_sec is not None else 0,
    "test_num_sonnets": len(test_generated) if test_sec is not None else 0,
    "dev_output": str(exp_dir / "generated_dev.txt") if dev_sec is not None else None,
    "test_output": str(exp_dir / "generated_test.txt") if test_sec is not None else None,
    "log_path": str(log_path),
  }
  (exp_dir / "metrics.json").write_text(
    json.dumps(metrics, indent=2, ensure_ascii=False),
    encoding="utf-8",
  )
  if dev_sec is not None:
    add_metadata_header(exp_dir / "generated_dev.txt", {**metadata, "split": "dev"})
  if test_sec is not None:
    add_metadata_header(exp_dir / "generated_test.txt", {**metadata, "split": "test"})

  if args.copy_checkpoint:
    shutil.copy2(checkpoint, exp_dir / "used_checkpoint.pt")

  print(
    f"finished {args.experiment}: "
    f"dev_sec={dev_sec if dev_sec is not None else 'skip'}, "
    f"test_sec={test_sec if test_sec is not None else 'skip'}, output={exp_dir}"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
  parser.add_argument("--checkpoint", default="checkpoints/p3_selected_checkpoint.pt")
  parser.add_argument("--output-root", default="../p3_decoding_results")
  parser.add_argument("--use-gpu", action="store_true")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--max-length", type=int, default=180)
  parser.add_argument("--target-lines", type=int, default=14)
  parser.add_argument("--split", choices=["dev", "test", "both"], default="both")
  parser.add_argument("--copy-checkpoint", action="store_true")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  os.chdir(ROOT)
  args.checkpoint = Path(args.checkpoint)
  if not args.checkpoint.is_absolute():
    args.checkpoint = ROOT / args.checkpoint
  args.output_root = Path(args.output_root)
  if not args.output_root.is_absolute():
    args.output_root = ROOT / args.output_root

  run_experiment(args)


if __name__ == "__main__":
  main()
