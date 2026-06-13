'''
소넷 생성을 위한 시작 코드.

실행:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
SonnetGPT 모델을 훈련하고, 필요한 제출용 파일을 작성한다.
'''

import time
from datetime import datetime

import os
import argparse
import random
import sys
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import SonnetsDataset
from models.gpt2 import GPT2Model
from optimizer import AdamW
from ridge_reranker import (
  chrf_target,
  extract_features,
  features_to_vector,
  fit_ridge_reranker,
  load_ridge_model,
  rerank_sonnets,
  save_ridge_fit,
)

from rhyme_decoding import generate_rhyming_sonnet, NUM_LINES

TQDM_DISABLE = False
LOG_TO_CONSOLE = True


# 로그 기록용 함수.
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


def init_training_log(args, model, log_path):
  """
  Prefix-tuning 실험 설정과 모델 파라미터 수를 로그 파일에 기록한다.
  """
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
    f.write("SonnetGPT Prefix-Tuning Training Log\n")
    f.write("====================================\n")
    f.write(f"started_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"model_size: {args.model_size}\n")
    f.write(f"epochs: {args.epochs}\n")
    f.write(f"batch_size: {args.batch_size}\n")
    f.write(f"learning_rate: {args.lr}\n")
    f.write(f"seed: {args.seed}\n")
    f.write(f"use_gpu: {args.use_gpu}\n")
    f.write(f"device: {next(model.parameters()).device}\n")
    f.write("\n")

    f.write("[Prefix-Tuning]\n")
    f.write(f"prefix_length: {args.prefix_length}\n")
    f.write(f"prefix_hidden_size: {args.prefix_hidden_size}\n")
    f.write(f"prefix_dropout: {args.prefix_dropout}\n")
    f.write("\n")

    f.write("[Generation]\n")
    f.write(f"temperature: {args.temperature}\n")
    f.write(f"top_p: {args.top_p}\n")
    f.write(f"max_length: {args.max_length}\n")
    f.write(f"line_level_num_candidates: {getattr(args, 'num_candidates', 10)}\n")
    f.write(f"rerank_candidates: {getattr(args, 'rerank_candidates', 1)}\n")
    f.write(f"reranker: {getattr(args, 'reranker', 'ridge')}\n")
    f.write(f"ridge_alpha: {getattr(args, 'ridge_alpha', 1.0)}\n")
    f.write(f"ridge_train_candidates: {getattr(args, 'ridge_train_candidates', 5)}\n")
    f.write(f"ridge_model_path: {getattr(args, 'ridge_model_path', None)}\n")
    f.write("\n")

    f.write("[Data]\n")
    f.write(f"sonnet_path: {args.sonnet_path}\n")
    f.write(f"dev_sonnet_path: {args.dev_sonnet_path}\n")
    f.write(f"held_out_sonnet_path: {args.held_out_sonnet_path}\n")
    f.write(f"sonnet_out: {args.sonnet_out}\n")
    f.write("\n")

    f.write("[Parameters]\n")
    f.write(f"total_params: {total_params:,}\n")
    f.write(f"trainable_params: {trainable_params:,}\n")
    f.write(f"frozen_params: {frozen_params:,}\n")
    f.write(f"trainable_ratio: {trainable_ratio:.6f}%\n")
    f.write("\n")

    f.write("[Epoch Logs]\n")


# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class PrefixEncoder(nn.Module):
  """
  Prefix-tuning용 trainable prefix module.

  작은 prefix embedding P'를 MLP에 통과시켜
  각 GPT layer의 prefix key/value를 만든다.

  반환:
    list length = num_layers
    각 원소: (prefix_key, prefix_value)
      prefix_key/value shape:
      [batch_size, num_heads, prefix_length, head_dim]
  """

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

    prefix_tokens = torch.arange(args.prefix_length).long()
    self.register_buffer("prefix_tokens", prefix_tokens, persistent=False)

  def forward(self, batch_size, device):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)

    # [batch_size, prefix_length, prefix_hidden_size]
    prefix_embeds = self.prefix_embedding(prefix_tokens)

    # [batch_size, prefix_length, num_layers * 2 * hidden_size]
    past_key_values = self.mlp(prefix_embeds)
    past_key_values = self.dropout(past_key_values)

    # [batch_size, prefix_length, num_layers, 2, num_heads, head_dim]
    past_key_values = past_key_values.view(
      batch_size,
      self.prefix_length,
      self.num_layers,
      2,
      self.num_heads,
      self.head_dim,
    )

    # [num_layers, 2, batch_size, num_heads, prefix_length, head_dim]
    past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5).contiguous()

    prefix_key_values = []
    for layer_idx in range(self.num_layers):
      prefix_key = past_key_values[layer_idx, 0]
      prefix_value = past_key_values[layer_idx, 1]
      prefix_key_values.append((prefix_key, prefix_value))

    return prefix_key_values


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 GPT-2 + Prefix-tuning 모델."""

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

    # Prefix-tuning: GPT-2 본체는 freeze하고 prefix만 학습한다.
    for param in self.gpt.parameters():
      param.requires_grad = False

    self.prefix_encoder = PrefixEncoder(args)

  def forward(self, input_ids, attention_mask):
    """
    시퀀스의 각 토큰 위치에 대한 next-token logits를 반환한다.
    """
    prefix_key_values = self.prefix_encoder(
      batch_size=input_ids.size(0),
      device=input_ids.device,
    )

    outputs = self.gpt(
      input_ids=input_ids,
      attention_mask=attention_mask,
      prefix_key_values=prefix_key_values,
    )

    hidden_states = outputs["last_hidden_state"]
    logits = self.gpt.hidden_state_to_token(hidden_states)

    return logits

  def get_device(self):
    return next(self.parameters()).device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    top-p sampling과 softmax temperature를 사용하여 prompt 뒤 continuation을 생성한다.

    반환:
      token_ids: prompt + generated tokens
      generated_output: prompt를 제외한 generated continuation 문자열
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    prompt_len = token_ids.shape[1]

    for _ in range(max_length):
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature

      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

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

    generated_ids = token_ids[0, prompt_len:].cpu().numpy().tolist()
    generated_output = self.tokenizer.decode(generated_ids)

    return token_ids, generated_output


def compute_lm_loss(model, batch, device):
  """
  train/dev 공통 LM loss 계산 함수.
  padding token 위치는 attention_mask로 제외한다.

  pad_token_id == eos_token_id인 GPT-2 설정에서도
  실제 EOS 토큰까지 ignore하지 않도록 attention_mask만 사용한다.
  """
  b_ids = batch["token_ids"].to(device)
  b_mask = batch["attention_mask"].to(device)

  logits = model(b_ids, b_mask)

  shift_logits = logits[:, :-1].contiguous()
  shift_labels = b_ids[:, 1:].contiguous()
  shift_mask = b_mask[:, 1:].contiguous()

  shift_logits = rearrange(shift_logits, "b t d -> (b t) d")
  shift_labels = shift_labels.flatten()
  shift_mask = shift_mask.flatten().float()

  token_losses = F.cross_entropy(
    shift_logits,
    shift_labels,
    reduction="none",
  )

  loss = (token_losses * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)
  return loss


@torch.no_grad()
def evaluate_lm_loss(model, dataloader, device):
  """
  dev set에서 gradient 없이 평균 LM loss 계산.
  """
  model.eval()

  total_loss = 0.0
  num_batches = 0

  for batch in tqdm(dataloader, desc="dev-eval", disable=TQDM_DISABLE):
    loss = compute_lm_loss(model, batch, device)
    total_loss += loss.item()
    num_batches += 1

  return total_loss / max(num_batches, 1)


@torch.no_grad()
def generate_sonnet_candidates(model, prompt_text, num_full_candidates, args):
  """
  prompt에 대해 완성 소네트 후보 여러 개를 생성한다.
  """
  candidates = []
  num_full_candidates = max(1, num_full_candidates)
  num_line_candidates = getattr(args, "num_candidates", 10)

  for _ in range(num_full_candidates):
    result = generate_rhyming_sonnet(
      model,
      prompt_text,
      num_candidates=num_line_candidates,
      temperature=args.temperature,
      top_p=args.top_p,
      max_line_tokens=22,
      min_line_tokens=4,
      soft_target_tokens=8, nl_boost=2.0,
      penalize_identical=False,
      verbose=False,
    )

    sonnet_lines = [line for line in result["text"].split("\n") if line.strip()][:NUM_LINES]
    candidates.append("\n".join(sonnet_lines))

  return candidates


@torch.no_grad()
def train_ridge_reranker(model, args):
  """
  dev prompt에서 후보를 만들고 gold와의 chrF를 target으로 Ridge reranker를 학습한다.
  """
  prompt_dataset = SonnetsDataset(args.ridge_prompt_path)
  gold_dataset = SonnetsDataset(args.ridge_gold_path)
  num_prompts = min(len(prompt_dataset), len(gold_dataset))

  if num_prompts == 0:
    raise ValueError("Ridge reranker needs at least one dev prompt/gold pair.")

  feature_rows = []
  targets = []
  train_candidates = max(1, args.ridge_train_candidates)

  for idx in tqdm(range(num_prompts), desc="ridge-train", disable=TQDM_DISABLE):
    _, prompt_text = prompt_dataset[idx]
    _, gold_text = gold_dataset[idx]

    prompt_start = time.time()
    write_log(
      f"Ridge training candidates for dev prompt {idx + 1}/{num_prompts}...",
      args.log_path,
    )

    candidate_sonnets = generate_sonnet_candidates(
      model,
      prompt_text,
      train_candidates,
      args,
    )

    for candidate in candidate_sonnets:
      features = extract_features(candidate, prompt=prompt_text)
      feature_rows.append(features_to_vector(features))
      targets.append(chrf_target(candidate, gold_text))

    write_log(
      f"Finished Ridge dev prompt {idx + 1}/{num_prompts} "
      f"in {time.time() - prompt_start:.2f}s",
      args.log_path,
    )

  return fit_ridge_reranker(
    feature_rows,
    targets,
    alpha=args.ridge_alpha,
  )


def save_model(model, optimizer, args, filepath):
  checkpoint_dir = os.path.dirname(filepath)
  if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)

  save_info = {
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "args": args,
    "system_rng": random.getstate(),
    "numpy_rng": np.random.get_state(),
    "torch_rng": torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  write_log(f"save the model to {filepath}", args.log_path)


def train(args):
  """Sonnet 데이터셋에서 prefix-tuning으로 GPT-2 훈련."""
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

  args = add_arguments(args)

  model = SonnetGPT(args)
  model = model.to(device)

  init_training_log(args, model, args.log_path)

  # 데이터셋 및 데이터로더 생성.
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

  # held-out 데이터셋은 generation sanity check용.
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

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

    log_msg = (
      f"Epoch {epoch}: "
      f"train_loss={train_loss:.4f}, "
      f"dev_loss={dev_loss:.4f}, "
      f"num_batches={num_batches}, "
      f"train+dev_elapsed_sec={epoch_time:.2f}, "
      f"dev_elapsed_sec={dev_time:.2f}"
    )
    write_log(log_msg, args.log_path)

    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      best_epoch = epoch
      save_model(model, optimizer, args, args.filepath)
      write_log(
        f"New best model saved at epoch {epoch} with dev loss {dev_loss:.4f}",
        args.log_path,
      )
    else:
      write_log(
        f"Model not saved. Best epoch: {best_epoch}, best dev loss: {best_dev_loss:.4f}",
        args.log_path,
      )

    if args.generate_each_epoch:
      write_log("Generating several output sonnets...", args.log_path)
      model.eval()

      for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        prompt_text = batch[1]
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
        )

        write_log(f"Sonnet {sonnet_id}\n{prompt_text}{continuation}\n", args.log_path)

    write_log("-" * 60, args.log_path)

  write_log(
    f"Training finished. Best epoch: {best_epoch}, best dev loss: {best_dev_loss:.4f}",
    args.log_path,
  )


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

  # dev loss 기준으로 저장된 best checkpoint 불러오기
  saved = torch.load(args.filepath, weights_only=False)

  model = SonnetGPT(saved["args"])
  model.load_state_dict(saved["model"])
  model = model.to(device)
  model.eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  rerank_candidates = max(1, getattr(args, "rerank_candidates", 1))
  use_ridge_reranker = args.reranker == "ridge" and rerank_candidates > 1

  ridge_fit = None
  ridge_model = None
  if use_ridge_reranker:
    ridge_model_path = getattr(args, "ridge_model_path", None)
    if ridge_model_path and os.path.exists(ridge_model_path):
      ridge_model = load_ridge_model(ridge_model_path)
      write_log(f"Loaded Ridge reranker from {ridge_model_path}", args.log_path)
    else:
      write_log(
        f"Training Ridge reranker from {args.ridge_prompt_path} / {args.ridge_gold_path} "
        f"with {args.ridge_train_candidates} candidates per prompt.",
        args.log_path,
      )

      if os.path.abspath(args.held_out_sonnet_path) == os.path.abspath(args.ridge_prompt_path):
        write_log(
          "Warning: held_out_sonnet_path is the same as ridge_prompt_path; "
          "Ridge selection will be optimistic on this split.",
          args.log_path,
        )

      ridge_fit = train_ridge_reranker(model, args)
      ridge_model = ridge_fit.model

      if ridge_model_path:
        save_ridge_fit(ridge_model_path, ridge_fit)
        write_log(f"Saved Ridge reranker to {ridge_model_path}", args.log_path)

      write_log(
        f"Ridge reranker trained: examples={ridge_fit.num_examples}, "
        f"target_mean={ridge_fit.target_mean:.4f}, target_std={ridge_fit.target_std:.4f}, "
        f"train_mae={ridge_fit.train_mae:.4f}, intercept={ridge_fit.model.intercept:.4f}",
        args.log_path,
      )

    coef_log = ", ".join(
      f"{name}={coef:.4f}"
      for name, coef in zip(ridge_model.feature_names, ridge_model.coef)
    )
    write_log(f"Ridge coefficients: {coef_log}", args.log_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    prompt_text = batch[1]

    candidate_count = rerank_candidates if use_ridge_reranker else 1
    generation_start = time.time()
    write_log(
      f"Generating {candidate_count} candidate(s) for Sonnet {sonnet_id}...",
      args.log_path,
    )

    candidate_sonnets = generate_sonnet_candidates(
      model,
      prompt_text,
      candidate_count,
      args,
    )

    if use_ridge_reranker:
      reranked = rerank_sonnets(
        candidate_sonnets,
        prompt=prompt_text,
        model=ridge_model,
      )
      full_sonnet = reranked.text
      feature_log = ", ".join(
        f"{name}={value:.3f}" for name, value in sorted(reranked.features.items())
      )
      write_log(
        f"Ridge reranker selected candidate {reranked.candidate_index + 1}/{rerank_candidates} "
        f"for Sonnet {sonnet_id}: predicted_chrf={reranked.score:.4f} ({feature_log})",
        args.log_path,
      )
    else:
      full_sonnet = candidate_sonnets[0]

    generated_sonnets.append((sonnet_id, full_sonnet))
    write_log(
      f"Finished Sonnet {sonnet_id} in {time.time() - generation_start:.2f}s",
      args.log_path,
    )
    write_log(f"Sonnet {sonnet_id}\n{full_sonnet}\n", args.log_path)
  
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
  parser.add_argument(
    "--filepath",
    type=str,
    default=None,
    help="Checkpoint path. Defaults to {epochs}-{lr}-prefix-sonnet.pt.",
  )

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action="store_true")

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument(
    "--top_p",
    type=float,
    help="Cumulative probability distribution for nucleus sampling.",
    default=0.9,
  )
  parser.add_argument("--max_length", type=int, default=128)
  parser.add_argument(
    "--num_candidates",
    type=int,
    default=10,
    help="Line-level candidates used inside rhyme-aware decoding.",
  )
  parser.add_argument(
    "--rerank_candidates",
    type=int,
    default=1,
    help="Number of full-sonnet candidates to generate and rerank. 1 disables final reranking.",
  )
  parser.add_argument(
    "--reranker",
    type=str,
    choices=["none", "ridge"],
    default="ridge",
    help="Final full-sonnet reranker to use when rerank_candidates > 1.",
  )
  parser.add_argument(
    "--ridge_alpha",
    type=float,
    default=1.0,
    help="L2 regularization strength for the Ridge reranker.",
  )
  parser.add_argument(
    "--ridge_train_candidates",
    type=int,
    default=5,
    help="Full-sonnet candidates generated per dev prompt to train the Ridge reranker.",
  )
  parser.add_argument(
    "--ridge_prompt_path",
    type=str,
    default="data/sonnets_held_out_dev.txt",
    help="Dev prompt file used to train the Ridge reranker.",
  )
  parser.add_argument(
    "--ridge_gold_path",
    type=str,
    default="data/TRUE_sonnets_held_out_dev.txt",
    help="Dev gold sonnet file used as chrF target for the Ridge reranker.",
  )
  parser.add_argument(
    "--ridge_model_path",
    type=str,
    default=None,
    help="Path to load/save the fitted Ridge reranker JSON. Existing files are loaded.",
  )

  parser.add_argument("--batch_size", help="The training batch size.", type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument(
    "--model_size",
    type=str,
    help="The model size as specified on hugging face.",
    choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    default="gpt2",
  )

  parser.add_argument("--log_path", type=str, default="logs/sonnet_train.log")
  parser.add_argument(
    "--log_to_file_only",
    action="store_true",
    help="Write logs only to log_path and disable tqdm console progress bars.",
  )

  # Prefix-tuning parameters.
  parser.add_argument("--prefix_length", type=int, default=10)
  parser.add_argument("--prefix_hidden_size", type=int, default=512)
  parser.add_argument("--prefix_dropout", type=float, default=0.1)

  # Logging / debugging.
  parser.add_argument("--log_every", type=int, default=20)
  parser.add_argument(
    "--generate_each_epoch",
    action="store_true",
    help="Generate held-out samples after every epoch. Useful for debugging, but slower and logs a lot.",
  )

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
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
  else:
    raise Exception(f"{args.model_size} is not supported.")

  return args


if __name__ == "__main__":
  args = get_args()
  LOG_TO_CONSOLE = not args.log_to_file_only
  TQDM_DISABLE = args.log_to_file_only
  if args.filepath is None:
    args.filepath = f"{args.epochs}-{args.lr}-prefix-sonnet.pt"

  seed_everything(args.seed)

  train(args)
  generate_submission_sonnets(args)
