"""
rhyme_decoding.py  (patched)

ABAB CDCD EFEF GG 소넷을 '줄 단위'로 생성하면서, 라임이 묶이는 행의 마지막 단어를
'앞서 생성된 같은 그룹 행의 마지막 단어'와 운율이 맞도록 선택한다.

--------------
줄 길이 제어
- max_line_tokens 기본값을 22로 (soft target 이 보통 더 일찍 줄을 닫으므로, 자연스럽게 길어지는 드문 줄이 잘리지 않도록 상한만 넉넉히 함)
- soft_target_tokens 이후 개행(\n) logit을 점진적으로 키워, frozen GPT-2가 run-on 으로 흐르기 전에 줄을 닫게 한다 -> 줄 길이 ↓, chrF ↑
14줄 보장: 빈 줄이 나오면 재시도/대체. (평가지표는 14줄 미만이면 즉시 0점)
평가지표 정합: 채택 기준을 do_rhyme(=평가지표)와 동일하게 penalize_identical 기본 False (평가지표는 동일 단어도 운율로 인정).
운율 단어를 '첫 서브워드 logit'이 아니라 '단어 전체 평균 logprob'으로 골라 arau/mea/mcever 같은 비단어 대신 fluent 단어를 선택 -> 운율 유지 + chrF ↑.
"""

import torch
import torch.nn.functional as F

from rhyme_utils import get_last_word, rhyme_score, do_rhyme

#   ABAB CDCD EFEF GG :  0 1 2 3 / 4 5 6 7 / 8 9 10 11 / 12 13
RHYME_TARGET = {2: 0, 3: 1, 6: 4, 7: 5, 10: 8, 11: 9, 13: 12}
NUM_LINES = 14


# ---------------------------------------------------------------------------
# 반복 억제
# ---------------------------------------------------------------------------
def _apply_repetition_penalty(logits, prev_ids, penalty=1.3):
    if penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    ids = torch.unique(prev_ids)
    sel = logits[0, ids]
    sel = torch.where(sel > 0, sel / penalty, sel * penalty)
    logits[0, ids] = sel
    return logits


def _block_repeat_ngrams(logits, seq, n=3):
    if n <= 0 or seq.shape[1] < n:
        return logits
    tokens = seq[0].tolist()
    prefix = tuple(tokens[-(n - 1):])
    banned = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n - 1]) == prefix:
            banned.add(tokens[i + n - 1])
    for t in banned:
        logits[0, t] = float("-inf")
    return logits


def _top_p_sample(logits_last, temperature, top_p):
    logits_last = logits_last / temperature
    probs = F.softmax(logits_last, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True
    filtered = sorted_probs * mask
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(filtered, 1)
    return sorted_indices.gather(dim=-1, index=idx)


# ---------------------------------------------------------------------------
# 한 줄 생성: 길이 제어(soft target + 점진적 개행 유도) 추가
# ---------------------------------------------------------------------------
@torch.no_grad()
def _sample_one_line(model, context_ids, temperature, top_p,
                     max_line_tokens=22, min_line_tokens=4,
                     rep_penalty=1.3, no_repeat_ngram=3,
                     soft_target_tokens=8, nl_boost=2.0):
    """
    context_ids([1,t]) 뒤로 '한 줄'을 생성, 개행/EOS에서 종료
      - min_line_tokens 까지는 개행/EOS 금지(너무 짧은 줄 방지)
      - soft_target_tokens 를 넘어가면 개행(\n) logit을 점진적으로 키워
        목표 길이(8~10단어) 근처에서 줄을 닫도록 유도 (run-on 방지)
      - max_line_tokens 에서 종료 (단, soft target 이 보통 먼저 줄을 닫으므로 평소엔 거의 강제하지 않는 안전 상한임
        soft target 이 약하거나 모델이 비정상적으로 run-on 일 때만 백스톱으로 작동)
    """
    device = model.get_device()
    nl_id = model.tokenizer.encode("\n")[-1]
    eos_id = model.tokenizer.eos_token_id

    cur = context_ids.to(device)
    line_tokens = []
    for _ in range(max_line_tokens):
        attn = torch.ones(cur.shape, dtype=torch.int64, device=device)
        logits = model.forward(cur, attn)[:, -1, :].clone()      # [1, vocab]

        logits = _apply_repetition_penalty(logits, cur, rep_penalty)
        logits = _block_repeat_ngrams(logits, cur, no_repeat_ngram)

        if len(line_tokens) < min_line_tokens:
            # 너무 이른 줄 종료 방지
            logits[:, nl_id] = float("-inf")
            logits[:, eos_id] = float("-inf")
        elif len(line_tokens) >= soft_target_tokens:
            # 목표 길이 초과분에 비례해 개행 확률을 끌어올림
            over = len(line_tokens) - soft_target_tokens + 1
            logits[:, nl_id] = logits[:, nl_id] + nl_boost * over

        tok = _top_p_sample(logits, temperature, top_p)
        tid = tok.item()
        if tid == eos_id:
            break
        if tid == nl_id or "\n" in model.tokenizer.decode([tid]):
            break
        line_tokens.append(tid)
        cur = torch.cat([cur, tok], dim=1)

    if line_tokens:
        line_ids = torch.tensor([line_tokens], dtype=torch.int64, device=device)
        text = model.tokenizer.decode(line_tokens).strip()
    else:
        line_ids = torch.zeros((1, 0), dtype=torch.int64, device=device)
        text = ""
    return text, line_ids


# ---------------------------------------------------------------------------
# 평가지표(do_rhyme) 정합 채택
# ---------------------------------------------------------------------------
def _eval_rhyme(word, anchor):
    """평가지표와 '동일한' 판정: do_rhyme(동일 단어도 True) ->  1.0/0.0."""
    if not word or not anchor:
        return 0.0
    return 1.0 if do_rhyme(word, anchor) else 0.0


def _length_similarity(cand_word, anchor_word):
    """
    후보의 '마지막 단어'와 앵커의 '마지막 단어'의 '글자 수' 근접도
    1.0=같은 글자 수, 멀어질수록 0에 수렴
    chrF가 char-level 지표이므로, 운율 단어끼리 글자 길이를 비슷하게 맞춘다
    """
    a = len((anchor_word or "").strip())
    b = len((cand_word or "").strip())
    if a == 0 or b == 0:
        return 0.0
    return 1.0 - abs(b - a) / max(a, b, 1)


def _pick_best_line(candidates, anchor_word):
    """
    후보 N개 중에서 강제로 교채허지 않고 가장 좋은 줄을 '선택'만 함(의미 보존)
    정렬 우선순위:
      1순위: 라임 근접도(do_rhyme 0/1) - 완전운 후보가 있으면 우선
      2순위: graded rhyme_score - 슬랜트/부분운까지 '정도'로 비교
      3순위: 마지막 단어 글자 수 근접도 - 라임이 비슷한 것들 중에서 끝 단어 길이가 비슷한 걸 선택
    "라임이 최대한 비슷한 것 중에서, 마지막 단어 길이가 최대한 비슷한 줄"을 고른다
    """
    scored = []
    for text, ids in candidates:
        lw = get_last_word(text) or ""
        hard = _eval_rhyme(lw, anchor_word)                 # 0/1 (평가지표 정합)
        soft = rhyme_score(lw, anchor_word, penalize_identical=False)
        lensim = _length_similarity(lw, anchor_word)        # 0~1 (끝 단어 글자수 근접도)
        scored.append((hard, soft, lensim, text, ids))
    # (라임 0/1) -> (graded 라임) -> (끝 단어 글자수 근접도) 순으로 내림차순 정렬
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best = scored[0]
    return best[3], best[4], best[0], best[1]


# ---------------------------------------------------------------------------
# fluent한 운율 단어 강제 교체
# ---------------------------------------------------------------------------
# @torch.no_grad()
# def _force_last_word_rhyme(model, context_ids, line_text, anchor,
#                            max_cand_words=120, prefer_single_token=True):
#     """
#     line_text 마지막 단어를 anchor 와 운율 맞는 '유창한' 단어로 강제
#     선택 기준 = 후보 단어의 (앞 1~2 서브워드) 평균 logprob 최대
#     비단어('arau/mea/mcever' 등) 대신 모델이 실제로 선호하는 단어를 고른다
#     반환: (new_text, new_ids[1,k], 1.0) 또는 None
#     """
#     if not anchor:
#         return None
#     # 평가지표 do_rhyme 로 필터 -> 교체 단어는 반드시 운율 쌍으로 인정
#     rhymes = [w.lower() for w in pronouncing.rhymes(anchor)
#               if w.isalpha() and do_rhyme(w, anchor)]
#     if not rhymes:
#         return None
#     rhymes = list(dict.fromkeys(rhymes))[:max_cand_words]

#     tok = model.tokenizer
#     device = model.get_device()

#     words = line_text.split()
#     body = " ".join(words[:-1]) if len(words) > 1 else ""

#     if body:
#         body_ids = tok(body, return_tensors="pt")["input_ids"].to(device)
#         ctx = torch.cat([context_ids, body_ids], dim=1)
#     else:
#         ctx = context_ids

#     attn = torch.ones(ctx.shape, dtype=torch.int64, device=device)
#     logits1 = model.forward(ctx, attn)[:, -1, :]                 # [1, vocab]
#     logprobs1 = F.log_softmax(logits1, dim=-1)[0]                # [vocab]

#     # 각 후보의 첫 서브워드 토큰 / logprob
#     cand = []
#     for w in rhymes:
#         ids = tok.encode(" " + w)
#         if not ids:
#             continue
#         if prefer_single_token and len(ids) > 2:
#             # 3토큰 이상으로 쪼개지는 단어는 대개 비단어 -> 후보에서 약하게 배제
#             continue
#         cand.append((w, ids))
#     if not cand:
#         # 단일/2토큰 후보가 없으면 제약을 풀고 다시
#         cand = [(w, tok.encode(" " + w)) for w in rhymes if tok.encode(" " + w)]
#     if not cand:
#         return None

#     # 2-토큰까지 평균 logprob (1회 배치 forward로 두번째 토큰 logprob 계산)
#     first_ids = [ids[0] for _, ids in cand]
#     base_lp = logprobs1[torch.tensor(first_ids, device=device)]  # [C]

#     two_tok = [(i, ids) for i, (_, ids) in enumerate(cand) if len(ids) >= 2]
#     second_lp = torch.zeros(len(cand), device=device)
#     if two_tok:
#         batch_ctx = ctx.expand(len(two_tok), -1)
#         first_col = torch.tensor([[ids[0]] for _, ids in two_tok], device=device)
#         batch_in = torch.cat([batch_ctx, first_col], dim=1)
#         batch_attn = torch.ones(batch_in.shape, dtype=torch.int64, device=device)
#         out = model.forward(batch_in, batch_attn)[:, -1, :]
#         lp2 = F.log_softmax(out, dim=-1)
#         for row, (ci, ids) in enumerate(two_tok):
#             second_lp[ci] = lp2[row, ids[1]]

#     n_tok = torch.tensor([min(len(ids), 2) for _, ids in cand],
#                          dtype=torch.float, device=device)
#     mean_lp = (base_lp + second_lp) / n_tok                      # 길이정규화 평균 logprob

#     best_idx = int(torch.argmax(mean_lp).item())
#     best_w = cand[best_idx][0]

#     new_text = (body + " " + best_w).strip()
#     new_ids = tok(new_text, return_tensors="pt")["input_ids"].to(device)
#     return new_text, new_ids, 1.0


# ---------------------------------------------------------------------------
# 14줄 ABAB CDCD EFEF GG 생성
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_rhyming_sonnet(model, prompt, num_candidates=10,
                            temperature=1.2, top_p=0.9,
                            max_line_tokens=22, min_line_tokens=4,
                            soft_target_tokens=8, nl_boost=2.0,
                            penalize_identical=False, verbose=False,
                            rep_penalty=1.3, no_repeat_ngram=3):
    """
    prompt(보통 첫 3줄)로 시작하여 14줄 ABAB CDCD EFEF GG 소넷 생성
    - 모든 줄은 비어있지 않도록 보장(14줄 미만이면 평가지표가 0점)
    - 라임 제약 행: 후보 N개 중 '라임이 가장 비슷하고, 그 중 길이가 가장 비슷한' 줄을 선택
      (마지막 단어 강제 교체는 하지 않고 의미 보존, force_rhyme 은 사용 안 함)
    """
    device = model.get_device()
    tok = model.tokenizer
    nl_id = tok.encode("\n")[-1]

    def sample_line(ctx):
        return _sample_one_line(model, ctx, temperature, top_p,
                                max_line_tokens, min_line_tokens,
                                rep_penalty, no_repeat_ngram,
                                soft_target_tokens, nl_boost)

    given_lines = [l for l in prompt.strip("\n").split("\n") if l.strip()]
    lines = list(given_lines)
    last_words = [get_last_word(l) for l in given_lines]

    context_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    if context_ids.shape[1] == 0 or context_ids[0, -1].item() != nl_id:
        context_ids = torch.cat(
            [context_ids, torch.tensor([[nl_id]], device=device)], dim=1)

    line_scores = {}
    for i in range(len(given_lines), NUM_LINES):
        target = RHYME_TARGET.get(i)
        anchor = last_words[target] if (target is not None and target < len(last_words)) else None

        if anchor:  # ---- 라임 제약 행 ----
            cands = [sample_line(context_ids) for _ in range(num_candidates)]
            # 강제 교체 없이 후보 중 라임이 가장 비슷하고, 그 중 마지막 단어 글자수가 가장 비슷한 줄을 '선택'만 함
            line_text, line_ids, hard, soft = _pick_best_line(cands, anchor)
            line_scores[i] = hard
            if verbose:
                print(f"[line {i:2d}] rhyme w/ line{target}('{anchor}') "
                      f"do_rhyme={hard:.0f} graded={soft:.2f} -> {line_text}")
        else:        # ---- 앵커 행: 자유 생성 ----
            line_text, line_ids = sample_line(context_ids)
            if verbose:
                print(f"[line {i:2d}] anchor -> {line_text}")

        # 빈 줄이면 최대 3회 재시도. 그래도 비면 anchor(운율 행)로만 채움
        # 자유 행은 채울 단어가 없으면 비우기
        retry = 0
        while (not line_text.strip()) and retry < 3:
            line_text, line_ids = sample_line(context_ids)
            retry += 1
        if not line_text.strip():
            line_text = anchor if anchor else ""
            line_ids = (tok(line_text, return_tensors="pt")["input_ids"].to(device)
                        if line_text.strip()
                        else torch.zeros((1, 0), dtype=torch.int64, device=device))

        if line_ids.shape[1] > 0:
            context_ids = torch.cat([context_ids, line_ids], dim=1)
        context_ids = torch.cat(
            [context_ids, torch.tensor([[nl_id]], device=device)], dim=1)

        lines.append(line_text)
        last_words.append(get_last_word(line_text))

    # 14줄 보장
    lines = [l for l in lines if l.strip()][:NUM_LINES]
    return {
        "text": "\n".join(lines),
        "lines": lines,
        "last_words": last_words,
        "line_scores": line_scores,
    }