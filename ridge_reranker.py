"""
Ridge reranker for generated sonnets.

The reranker learns a small linear model over hand-built candidate features.
Training targets are automatic chrF scores against dev gold sonnets, so no
manual labeling is required.
"""

import math
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from rhyme_utils import do_rhyme, get_last_word, rhyme_score


NUM_LINES = 14
RHYME_PAIRS = (
  (0, 2), (1, 3),
  (4, 6), (5, 7),
  (8, 10), (9, 11),
  (12, 13),
)

FEATURE_NAMES = (
  "rhyme",
  "graded_rhyme",
  "length",
  "length_variance",
  "repetition",
  "ending_diversity",
  "prompt_overlap",
)


@dataclass
class RidgeModel:
  feature_names: Tuple[str, ...]
  coef: List[float]
  intercept: float
  alpha: float

  def predict_features(self, features: Dict[str, float]) -> float:
    values = np.array([features[name] for name in self.feature_names], dtype=np.float64)
    coef = np.array(self.coef, dtype=np.float64)
    return float(self.intercept + values @ coef)

  def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
    coef = np.array(self.coef, dtype=np.float64)
    return self.intercept + matrix @ coef


@dataclass
class RidgeFit:
  model: RidgeModel
  train_mae: float
  target_mean: float
  target_std: float
  num_examples: int

  def to_dict(self) -> Dict[str, object]:
    return {
      "feature_names": list(self.model.feature_names),
      "coef": self.model.coef,
      "intercept": self.model.intercept,
      "alpha": self.model.alpha,
      "train_mae": self.train_mae,
      "target_mean": self.target_mean,
      "target_std": self.target_std,
      "num_examples": self.num_examples,
    }


@dataclass
class RidgeRerankResult:
  text: str
  score: float
  features: Dict[str, float]
  candidate_index: int


def split_sonnet_lines(text: str, max_lines: int = NUM_LINES) -> List[str]:
  """Return non-empty sonnet lines, stripped and capped to max_lines."""
  return [line.strip() for line in text.split("\n") if line.strip()][:max_lines]


def normalize_sonnet_text(text: str) -> str:
  """Normalize a generated sonnet to the non-empty lines used for scoring."""
  return "\n".join(split_sonnet_lines(text))


def _word_tokens(text: str) -> List[str]:
  return re.findall(r"[A-Za-z']+", text.lower())


def _length_features(lines: Sequence[str], target_words: float = 8.0) -> Tuple[float, float]:
  if not lines:
    return 0.0, 0.0

  lengths = [len(_word_tokens(line)) for line in lines]
  closeness = [
    max(0.0, 1.0 - abs(length - target_words) / target_words)
    for length in lengths
  ]
  length_score = sum(closeness) / len(closeness)

  mean_len = sum(lengths) / len(lengths)
  variance = sum((length - mean_len) ** 2 for length in lengths) / len(lengths)
  variance_score = math.exp(-variance / 12.0)

  return length_score, variance_score


def _rhyme_features(lines: Sequence[str]) -> Tuple[float, float]:
  if len(lines) < NUM_LINES:
    return 0.0, 0.0

  hard_scores = []
  graded_scores = []
  for left, right in RHYME_PAIRS:
    left_word = get_last_word(lines[left])
    right_word = get_last_word(lines[right])

    hard_scores.append(
      1.0 if left_word and right_word and do_rhyme(left_word, right_word) else 0.0
    )
    graded_scores.append(
      rhyme_score(left_word, right_word, penalize_identical=False)
      if left_word and right_word else 0.0
    )

  return (
    sum(hard_scores) / len(RHYME_PAIRS),
    sum(graded_scores) / len(RHYME_PAIRS),
  )


def _repetition_feature(lines: Sequence[str], n: int = 3) -> float:
  words = _word_tokens(" ".join(lines))
  if len(words) < n:
    return 1.0

  ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
  if not ngrams:
    return 1.0

  duplicate_count = len(ngrams) - len(set(ngrams))
  duplicate_ratio = duplicate_count / len(ngrams)
  return max(0.0, 1.0 - duplicate_ratio * 6.0)


def _ending_diversity_feature(lines: Sequence[str]) -> float:
  endings = [get_last_word(line) for line in lines]
  endings = [word for word in endings if word]
  if not endings:
    return 0.0

  return len(set(endings)) / len(endings)


def _prompt_overlap_feature(lines: Sequence[str], prompt: str) -> float:
  if not prompt:
    return 1.0

  prompt_lines = split_sonnet_lines(prompt)
  if not prompt_lines:
    return 1.0

  generated_tail = lines[len(prompt_lines):]
  if not generated_tail:
    return 1.0

  prompt_words = set(_word_tokens(" ".join(prompt_lines)))
  if not prompt_words:
    return 1.0

  tail_words = _word_tokens(" ".join(generated_tail))
  if not tail_words:
    return 0.0

  overlap_ratio = sum(1 for word in tail_words if word in prompt_words) / len(tail_words)
  return max(0.0, 1.0 - overlap_ratio * 2.0)


def extract_features(text: str, prompt: str = "") -> Dict[str, float]:
  """Extract the feature vector used by the Ridge reranker."""
  lines = split_sonnet_lines(text)
  length_score, length_variance_score = _length_features(lines)
  hard_rhyme, graded_rhyme = _rhyme_features(lines)

  return {
    "rhyme": hard_rhyme,
    "graded_rhyme": graded_rhyme,
    "length": length_score,
    "length_variance": length_variance_score,
    "repetition": _repetition_feature(lines),
    "ending_diversity": _ending_diversity_feature(lines),
    "prompt_overlap": _prompt_overlap_feature(lines, prompt),
  }


def features_to_vector(features: Dict[str, float]) -> List[float]:
  """Convert a feature dict to a stable ordered vector."""
  return [features[name] for name in FEATURE_NAMES]


def chrf_target(candidate: str, gold: str) -> float:
  """Return normalized chrF in [0, 1] for candidate against one gold sonnet."""
  from sacrebleu.metrics import CHRF

  chrf = CHRF()
  score = chrf.corpus_score([candidate], [[gold]]).score
  return float(score / 100.0)


def fit_ridge_reranker(
    feature_rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    alpha: float = 1.0,
) -> RidgeFit:
  """Fit a small Ridge model with an unregularized intercept."""
  if not feature_rows:
    raise ValueError("fit_ridge_reranker requires at least one training example.")
  if len(feature_rows) != len(targets):
    raise ValueError("feature_rows and targets must have the same length.")

  x = np.asarray(feature_rows, dtype=np.float64)
  y = np.asarray(targets, dtype=np.float64)

  ones = np.ones((x.shape[0], 1), dtype=np.float64)
  x_aug = np.concatenate([ones, x], axis=1)

  penalty = np.eye(x_aug.shape[1], dtype=np.float64) * alpha
  penalty[0, 0] = 0.0

  lhs = x_aug.T @ x_aug + penalty
  rhs = x_aug.T @ y

  try:
    params = np.linalg.solve(lhs, rhs)
  except np.linalg.LinAlgError:
    params = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

  intercept = float(params[0])
  coef = [float(value) for value in params[1:]]
  model = RidgeModel(
    feature_names=FEATURE_NAMES,
    coef=coef,
    intercept=intercept,
    alpha=alpha,
  )

  predictions = model.predict_matrix(x)
  train_mae = float(np.mean(np.abs(predictions - y)))

  return RidgeFit(
    model=model,
    train_mae=train_mae,
    target_mean=float(np.mean(y)),
    target_std=float(np.std(y)),
    num_examples=len(targets),
  )


def save_ridge_fit(path: str, ridge_fit: RidgeFit):
  """Save a fitted Ridge reranker to JSON for later inference."""
  output_dir = os.path.dirname(path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  with open(path, "w", encoding="utf-8") as f:
    json.dump(ridge_fit.to_dict(), f, indent=2, ensure_ascii=False)


def load_ridge_model(path: str) -> RidgeModel:
  """Load a Ridge reranker model from JSON."""
  with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)

  return RidgeModel(
    feature_names=tuple(payload["feature_names"]),
    coef=[float(value) for value in payload["coef"]],
    intercept=float(payload["intercept"]),
    alpha=float(payload.get("alpha", 0.0)),
  )


def rerank_sonnets(
    candidates: Sequence[str],
    prompt: str,
    model: RidgeModel,
) -> RidgeRerankResult:
  """Select the candidate with the highest Ridge-predicted chrF."""
  if not candidates:
    raise ValueError("rerank_sonnets requires at least one candidate.")

  best = None
  for candidate_index, candidate in enumerate(candidates):
    normalized = normalize_sonnet_text(candidate)
    features = extract_features(normalized, prompt=prompt)
    score = model.predict_features(features)
    result = RidgeRerankResult(
      text=normalized,
      score=score,
      features=features,
      candidate_index=candidate_index,
    )
    if best is None or result.score > best.score:
      best = result

  return best
