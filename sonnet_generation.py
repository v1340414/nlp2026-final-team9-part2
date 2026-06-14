"""
Prefix-only SonnetGPT 학습 및 기본 생성 코드.

P3 checkpoint를 불러와 CMU/rerank decoding에 사용할 수 있도록 하는 핵심 모델 코드이다.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW


TQDM_DISABLE = False
LOG_TO_CONSOLE = True


def write_log(message, log_path):
  if LOG_TO_CONSOLE:
    try:
      print(message)
    except UnicodeEncodeError:
      encoding = sys.stdout.encoding or "utf-8"
      safe_message = message.encode(encoding, errors="replace").decode(encoding)
      print(safe_message)

  if log_path is not None:
    log_dir = os.path.dirname(log_path)
    if log_dir:
      os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
      f.write(message + "\n")


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def add_arguments(args):
  if args.model_size == "gpt2":
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == "gpt2-medium":
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == "gpt2-large":
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  elif args.model_size == "gpt2-xl":
    args.d = 1600
    args.l = 48
    args.num_heads = 25
  else:
    raise ValueError(f"{args.model_size}는 지원하지 않는 model_size입니다.")
  return args


class PrefixEncoder(nn.Module):
  """각 GPT layer에 주입할 trainable key/value prefix를 생성하는 모듈."""

  def __init__(self, args):
    super().__init__()
    self.prefix_length = args.prefix_length
    self.num_layers = args.l
    self.num_heads = args.num_heads
    self.hidden_size = args.d
    self.head_dim = args.d // args.num_heads

    self.prefix_embedding = nn.Embedding(args.prefix_length, args.prefix_hidden_size)
    self.mlp = nn.Sequential(
      nn.Linear(args.prefix_hidden_size, args.prefix_hidden_size),
      nn.Tanh(),
      nn.Linear(args.prefix_hidden_size, args.l * 2 * args.d),
    )
    self.dropout = nn.Dropout(args.prefix_dropout)
    self.register_buffer(
      "prefix_tokens",
      torch.arange(args.prefix_length).long(),
      persistent=False,
    )

  def forward(self, batch_size, device):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    prefix_embeds = self.prefix_embedding(prefix_tokens)
    past_key_values = self.dropout(self.mlp(prefix_embeds))
    past_key_values = past_key_values.view(
      batch_size,
      self.prefix_length,
      self.num_layers,
      2,
      self.num_heads,
      self.head_dim,
    )
    past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5).contiguous()
    return [
      (past_key_values[layer_idx, 0], past_key_values[layer_idx, 1])
      for layer_idx in range(self.num_layers)
    ]


class SonnetGPT(nn.Module):
  """base GPT-2 weight는 고정하고 prefix key/value vector만 사용하는 SonnetGPT."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(
      model=args.model_size,
      d=args.d,
      l=args.l,
      num_heads=args.num_heads,
    )
    self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    self.tokenizer.pad_token = self.tokenizer.eos_token

    for param in self.gpt.parameters():
      param.requires_grad = False

    self.prefix_encoder = PrefixEncoder(args)

  def forward(self, input_ids, attention_mask):
    prefix_key_values = self.prefix_encoder(
      batch_size=input_ids.size(0),
      device=input_ids.device,
    )
    outputs = self.gpt(
      input_ids=input_ids,
      attention_mask=attention_mask,
      prefix_key_values=prefix_key_values,
    )
    return self.gpt.hidden_state_to_token(outputs["last_hidden_state"])

  def get_device(self):
    return next(self.parameters()).device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.85, max_length=180, target_lines=None):
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    prompt_len = token_ids.shape[1]
    newline_token_id = self.tokenizer.encode("\n")[-1]

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature
      probs = torch.softmax(logits_last_token, dim=-1)

      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()
      top_p_mask[..., 0] = True

      filtered_probs = sorted_probs * top_p_mask
      filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [
          attention_mask,
          torch.ones((token_ids.size(0), 1), dtype=torch.int64).to(self.get_device()),
        ],
        dim=1,
      )

      if target_lines and sampled_token.item() == newline_token_id:
        decoded_text = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
        if len([line for line in decoded_text.splitlines() if line.strip()]) >= target_lines:
          break

    generated_ids = token_ids[0, prompt_len:].cpu().numpy().tolist()
    return token_ids, self.tokenizer.decode(generated_ids)


def init_training_log(args, model, log_path):
  if log_path is None:
    return

  log_dir = os.path.dirname(log_path)
  if log_dir:
    os.makedirs(log_dir, exist_ok=True)

  total_params = sum(p.numel() for p in model.parameters())
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  frozen_params = total_params - trainable_params
  trainable_ratio = 100.0 * trainable_params / total_params

  with open(log_path, "w", encoding="utf-8") as f:
    f.write("SonnetGPT Prefix-Only Training Log\n")
    f.write("==================================\n")
    f.write(f"started_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"model_size: {args.model_size}\n")
    f.write(f"epochs: {args.epochs}\n")
    f.write(f"batch_size: {args.batch_size}\n")
    f.write(f"learning_rate: {args.lr}\n")
    f.write(f"seed: {args.seed}\n")
    f.write(f"use_gpu: {args.use_gpu}\n")
    f.write(f"device: {next(model.parameters()).device}\n\n")
    f.write("[Prefix-Tuning]\n")
    f.write(f"prefix_length: {args.prefix_length}\n")
    f.write(f"prefix_hidden_size: {args.prefix_hidden_size}\n")
    f.write(f"prefix_dropout: {args.prefix_dropout}\n\n")
    f.write("[Early Stopping]\n")
    f.write(f"early_stop_patience: {args.early_stop_patience}\n")
    f.write(f"early_stop_min_delta: {args.early_stop_min_delta}\n\n")
    f.write("[Plain Generation]\n")
    f.write(f"temperature: {args.temperature}\n")
    f.write(f"top_p: {args.top_p}\n")
    f.write(f"max_length: {args.max_length}\n\n")
    f.write("[Data]\n")
    f.write(f"sonnet_path: {args.sonnet_path}\n")
    f.write(f"dev_sonnet_path: {args.dev_sonnet_path}\n")
    f.write(f"held_out_sonnet_path: {args.held_out_sonnet_path}\n")
    f.write(f"sonnet_out: {args.sonnet_out}\n\n")
    f.write("[Parameters]\n")
    f.write(f"total_params: {total_params:,}\n")
    f.write(f"trainable_params: {trainable_params:,}\n")
    f.write(f"frozen_params: {frozen_params:,}\n")
    f.write(f"trainable_ratio: {trainable_ratio:.6f}%\n\n")
    f.write("[Epoch Logs]\n")


def compute_lm_loss(model, batch, device):
  b_ids = batch["token_ids"].to(device)
  b_mask = batch["attention_mask"].to(device)
  logits = model(b_ids, b_mask)

  shift_logits = logits[:, :-1].contiguous()
  shift_labels = b_ids[:, 1:].contiguous()
  shift_mask = b_mask[:, 1:].contiguous()

  shift_logits = rearrange(shift_logits, "b t d -> (b t) d")
  shift_labels = shift_labels.flatten()
  shift_mask = shift_mask.flatten().float()

  token_losses = F.cross_entropy(shift_logits, shift_labels, reduction="none")
  return (token_losses * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_lm_loss(model, dataloader, device):
  model.eval()
  total_loss = 0.0
  num_batches = 0
  for batch in tqdm(dataloader, desc="dev-eval", disable=TQDM_DISABLE):
    loss = compute_lm_loss(model, batch, device)
    total_loss += loss.item()
    num_batches += 1
  return total_loss / max(num_batches, 1)


def save_model(model, args, filepath):
  checkpoint_dir = os.path.dirname(filepath)
  if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

  torch.save(
    {
      "checkpoint_type": "prefix_only",
      "prefix_encoder": model.prefix_encoder.state_dict(),
      "args": args,
      "system_rng": random.getstate(),
      "numpy_rng": np.random.get_state(),
      "torch_rng": torch.random.get_rng_state(),
    },
    filepath,
  )
  write_log(f"save prefix checkpoint to {filepath}", args.log_path)


def train(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
  args = add_arguments(args)

  model = SonnetGPT(args).to(device)
  init_training_log(args, model, args.log_path)

  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(
    sonnet_dataset,
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=sonnet_dataset.collate_fn,
  )
  dev_sonnet_dataset = SonnetsDataset(args.dev_sonnet_path)
  dev_sonnet_dataloader = DataLoader(
    dev_sonnet_dataset,
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=dev_sonnet_dataset.collate_fn,
  )

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = AdamW(trainable_params, lr=args.lr)

  total_params = sum(p.numel() for p in model.parameters())
  trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
  write_log(
    f"Trainable params: {trainable_param_count:,} / {total_params:,} "
    f"({100 * trainable_param_count / total_params:.4f}%)",
    args.log_path,
  )

  best_dev_loss = float("inf")
  best_epoch = -1
  early_stop_best_loss = float("inf")
  epochs_without_improvement = 0
  stopped_early = False
  early_stop_reason = None
  epoch_metrics = []
  top_k = max(0, int(getattr(args, "save_top_k_checkpoints", 0)))
  top_checkpoint_dir = getattr(args, "top_checkpoint_dir", None)
  top_checkpoints = []
  if top_k > 0 and top_checkpoint_dir:
    os.makedirs(top_checkpoint_dir, exist_ok=True)

  for epoch in range(args.epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(sonnet_dataloader, desc=f"train-{epoch}", disable=TQDM_DISABLE)):
      optimizer.zero_grad()
      loss = compute_lm_loss(model, batch, device)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1
      if args.log_every > 0 and (step + 1) % args.log_every == 0:
        write_log(
          f"Epoch {epoch} | Step {step + 1}/{len(sonnet_dataloader)} | "
          f"batch_loss={loss.item():.4f}",
          args.log_path,
        )

    train_loss = train_loss / max(num_batches, 1)
    dev_start_time = time.time()
    dev_loss = evaluate_lm_loss(model, dev_sonnet_dataloader, device)
    dev_time = time.time() - dev_start_time
    epoch_time = time.time() - epoch_start_time

    write_log(
      f"Epoch {epoch}: train_loss={train_loss:.4f}, dev_loss={dev_loss:.4f}, "
      f"num_batches={num_batches}, train+dev_elapsed_sec={epoch_time:.2f}, "
      f"dev_elapsed_sec={dev_time:.2f}",
      args.log_path,
    )

    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      best_epoch = epoch
      save_model(model, args, args.filepath)
      write_log(
        f"New best prefix checkpoint at epoch {epoch} with dev loss {dev_loss:.4f}",
        args.log_path,
      )

    improved_for_early_stop = dev_loss < (early_stop_best_loss - args.early_stop_min_delta)
    if improved_for_early_stop:
      early_stop_best_loss = dev_loss
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1

    epoch_record = {
      "epoch": epoch,
      "train_loss": train_loss,
      "dev_loss": dev_loss,
      "early_stop_best_loss": early_stop_best_loss,
      "epochs_without_improvement": epochs_without_improvement,
      "train_dev_elapsed_sec": epoch_time,
      "dev_elapsed_sec": dev_time,
      "checkpoint_path": None,
    }

    if top_k > 0 and top_checkpoint_dir:
      worst_dev_loss = max((row["dev_loss"] for row in top_checkpoints), default=float("inf"))
      should_save_top = len(top_checkpoints) < top_k or dev_loss < worst_dev_loss
      if should_save_top:
        checkpoint_path = os.path.join(top_checkpoint_dir, f"epoch_{epoch:02d}.pt")
        save_model(model, args, checkpoint_path)
        epoch_record["checkpoint_path"] = checkpoint_path
        top_checkpoints.append({
          "epoch": epoch,
          "dev_loss": dev_loss,
          "checkpoint_path": checkpoint_path,
        })
        top_checkpoints.sort(key=lambda row: row["dev_loss"])
        while len(top_checkpoints) > top_k:
          removed = top_checkpoints.pop()
          removed_path = removed["checkpoint_path"]
          if os.path.exists(removed_path) and os.path.abspath(removed_path) != os.path.abspath(args.filepath):
            os.remove(removed_path)
            write_log(f"Removed non-top checkpoint {removed_path}", args.log_path)

    epoch_metrics.append(epoch_record)
    write_log("-" * 60, args.log_path)

    if (
      args.early_stop_patience > 0
      and epochs_without_improvement >= args.early_stop_patience
    ):
      stopped_early = True
      early_stop_reason = (
        f"no dev loss improvement >= {args.early_stop_min_delta} "
        f"for {args.early_stop_patience} consecutive epoch(s)"
      )
      write_log(
        f"Early stopping at epoch {epoch}: {early_stop_reason}",
        args.log_path,
      )
      break

  write_log(
    f"Training finished. Best epoch: {best_epoch}, best dev loss: {best_dev_loss:.4f}",
    args.log_path,
  )

  metrics_path = getattr(args, "epoch_metrics_path", None)
  if metrics_path:
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
      os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
      json.dump(
        {
          "best_epoch": best_epoch,
          "best_dev_loss": best_dev_loss,
          "stopped_early": stopped_early,
          "stop_epoch": epoch_metrics[-1]["epoch"] if epoch_metrics else None,
          "early_stop_reason": early_stop_reason,
          "epoch_metrics": epoch_metrics,
          "top_checkpoints": top_checkpoints,
        },
        f,
        indent=2,
        ensure_ascii=False,
      )

  return {
    "best_epoch": best_epoch,
    "best_dev_loss": best_dev_loss,
    "stopped_early": stopped_early,
    "stop_epoch": epoch_metrics[-1]["epoch"] if epoch_metrics else None,
    "early_stop_reason": early_stop_reason,
    "epoch_metrics": epoch_metrics,
    "top_checkpoints": top_checkpoints,
  }


def load_sonnet_model_from_checkpoint(filepath, device):
  saved = torch.load(filepath, weights_only=False, map_location=device)
  model = SonnetGPT(saved["args"])
  model.prefix_encoder.load_state_dict(saved["prefix_encoder"])
  model = model.to(device)
  model.eval()
  return model


def keep_first_nonempty_lines(text, target_lines):
  if not target_lines:
    return text.strip()

  kept = []
  nonempty_count = 0
  for line in text.splitlines():
    if line.strip():
      nonempty_count += 1
    if nonempty_count <= target_lines:
      kept.append(line)
    if nonempty_count >= target_lines:
      break
  return "\n".join(kept).strip()


@torch.no_grad()
def generate_plain_submission_sonnets(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
  model = load_sonnet_model_from_checkpoint(args.filepath, device)
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for sonnet_id, prompt_text in held_out_sonnet_dataset:
    generation_start = time.time()
    write_log(f"Plain generation for Sonnet {sonnet_id}...", args.log_path)

    encoding = model.tokenizer(
      prompt_text,
      return_tensors="pt",
      padding=True,
      truncation=True,
    ).to(device)
    _, continuation = model.generate(
      encoding["input_ids"],
      temperature=args.temperature,
      top_p=args.top_p,
      max_length=args.max_length,
      target_lines=getattr(args, "target_lines", 14),
    )
    sonnet_text = keep_first_nonempty_lines(
      prompt_text + continuation,
      getattr(args, "target_lines", 14),
    )
    generated_sonnets.append((sonnet_id, sonnet_text))
    write_log(
      f"Finished Sonnet {sonnet_id} in {time.time() - generation_start:.2f}s",
      args.log_path,
    )

  output_dir = os.path.dirname(args.sonnet_out)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write("--Generated Sonnets-- \n\n")
    for sonnet_id, sonnet_text in generated_sonnets:
      f.write(f"\n{sonnet_id}\n")
      f.write(sonnet_text)

  return generated_sonnets


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--dev_sonnet_path", type=str, default="data/TRUE_sonnets_held_out_dev.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--filepath", type=str, default=None)
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_p", type=float, default=0.85)
  parser.add_argument("--max_length", type=int, default=180)
  parser.add_argument("--target_lines", type=int, default=14)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=2e-4)
  parser.add_argument(
    "--model_size",
    type=str,
    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    default="gpt2-xl",
  )
  parser.add_argument("--log_path", type=str, default="logs/prefix_only.log")
  parser.add_argument("--log_to_file_only", action="store_true")
  parser.add_argument("--prefix_length", type=int, default=10)
  parser.add_argument("--prefix_hidden_size", type=int, default=512)
  parser.add_argument("--prefix_dropout", type=float, default=0.1)
  parser.add_argument("--log_every", type=int, default=20)
  parser.add_argument("--early_stop_patience", type=int, default=2)
  parser.add_argument("--early_stop_min_delta", type=float, default=0.005)
  parser.add_argument("--save_top_k_checkpoints", type=int, default=0)
  parser.add_argument("--top_checkpoint_dir", type=str, default=None)
  parser.add_argument("--epoch_metrics_path", type=str, default=None)
  args = parser.parse_args()
  if args.filepath is None:
    args.filepath = f"{args.epochs}-{args.lr}-prefix-only.pt"
  return args


if __name__ == "__main__":
  args = get_args()
  LOG_TO_CONSOLE = not args.log_to_file_only
  TQDM_DISABLE = args.log_to_file_only
  seed_everything(args.seed)
  train(args)
  generate_plain_submission_sonnets(args)
