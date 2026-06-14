"""
CMU Pronouncing Dictionary를 사용한 운율(rhyme) 유틸리티.
행의 마지막 단어 운율 검사 및 소넷 운율 점수 계산.

[확장] 기존 함수(get_last_word/get_rhyme_phonemes/do_rhyme/check_sonnet_rhyme_scheme/
get_rhyming_candidates)는 그대로 두고, 아래에 '정도(0~1)' 기반 graded 함수를 추가했다.
디코딩 시점에 '앞서 생성된 마지막 단어와 가장 잘 맞는 후보'를 고를 때 사용한다.
"""

import re
import pronouncing


def get_last_word(line):
    """행의 마지막 단어 추출 (구두점 제거)."""
    words = line.strip().split()
    if not words:
        return None
    return words[-1].strip(".,;:!?\"'").lower()


def get_rhyme_phonemes(word):
    """CMU dict에서 단어의 운율 음소(마지막 강세 모음 이후)를 반환."""
    phones_list = pronouncing.phones_for_word(word.lower().strip(".,;:!?\"'"))
    if not phones_list:
        return None
    return pronouncing.rhyming_part(phones_list[0])


def do_rhyme(word1, word2):
    """두 단어의 운율이 일치하는지 확인 (binary)."""
    r1 = get_rhyme_phonemes(word1)
    r2 = get_rhyme_phonemes(word2)
    if r1 is None or r2 is None:
        return False
    return r1 == r2


def check_sonnet_rhyme_scheme(sonnet_text):
    """
    생성된 소넷의 운율 점수 계산.
    셰익스피어 소넷 패턴: ABAB CDCD EFEF GG
    반환: (맞는 운율 쌍 수, 전체 쌍 수, 점수)
    """
    lines = [l for l in sonnet_text.strip().split('\n') if l.strip()]
    if len(lines) < 14:
        return 0, 7, 0.0

    last_words = [get_last_word(line) for line in lines[:14]]

    rhyme_pairs = [
        (0, 2), (1, 3),   # ABAB
        (4, 6), (5, 7),   # CDCD
        (8, 10), (9, 11), # EFEF
        (12, 13),         # GG
    ]

    correct = sum(
        1 for i, j in rhyme_pairs
        if last_words[i] and last_words[j] and do_rhyme(last_words[i], last_words[j])
    )
    return correct, len(rhyme_pairs), correct / len(rhyme_pairs)


def get_rhyming_candidates(tokenizer, candidate_token_ids, target_word):
    """
    토큰 후보 목록 중 target_word와 운율이 맞는 토큰 인덱스 반환.
    generate() 내부에서 운율 강제 적용 시 사용.
    """
    for tok in candidate_token_ids:
        candidate_word = tokenizer.decode([tok]).strip().lower()
        if do_rhyme(candidate_word, target_word):
            return tok
    return None  # 운율 맞는 후보 없으면 None


# ===========================================================================
# [추가] Graded rhyme score : 운율 '일치 정도'(0.0~1.0)
# ===========================================================================

def _strip_stress(phones):
    """강세 숫자 제거: ['AY1','ER0'] -> ['AY','ER'] (부분운 비교용)."""
    return [re.sub(r'\d', '', p) for p in phones]


def _suffix_match(a, b):
    """두 시퀀스의 끝에서부터 연속 일치 개수."""
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            n += 1
        else:
            break
    return n


def rhyme_score(word1, word2, penalize_identical=True):
    """
    두 단어의 운율 '일치 정도'를 0.0~1.0 으로 반환.
      1.0  : 완전운 (라임 부분 음소열 동일)  예) day/way, fire/desire
      0~1  : 부분운/슬랜트 라임 (끝 음소 일치 비율) 예) love/move -> 0.5
      0.0  : 전혀 안 맞음, 또는 (penalize_identical=True 이고) 같은 단어
    CMUdict에 없는 단어는 철자 끝자리 일치로 근사(fallback).

    penalize_identical:
      True  -> 앵커와 '같은 단어'는 0점 처리(에코 방지, 더 좋은 소넷).
      False -> 같은 단어도 1.0 (check_sonnet_rhyme_scheme의 do_rhyme 기준에
               더 가깝게 맞추고 싶을 때).
    """
    w1 = (word1 or '').lower().strip(".,;:!?\"'")
    w2 = (word2 or '').lower().strip(".,;:!?\"'")
    if not w1 or not w2:
        return 0.0
    if penalize_identical and w1 == w2:
        return 0.0

    r1 = get_rhyme_phonemes(w1)
    r2 = get_rhyme_phonemes(w2)
    if r1 is None or r2 is None:            # 사전에 없음 -> 철자 fallback
        return _suffix_match(list(w1), list(w2)) / max(len(w1), len(w2))

    p1 = _strip_stress(r1.split())
    p2 = _strip_stress(r2.split())
    if p1 == p2:
        return 1.0
    return _suffix_match(p1, p2) / max(len(p1), len(p2))