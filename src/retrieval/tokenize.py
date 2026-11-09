"""BM25용 토큰화 후보 (STEP 5, 06_MODEL_SELECTION.md §5.3, DECISIONS.md D-035).

후보는 네 가지이며 어느 것을 쓸지는 development 질문으로 정한다(규칙은 실행 전에 고정).
- `whitespace`: 공백으로 나누고 문장부호를 떼어 낸 단어(기준선)
- `kiwi`: kiwipiepy 형태소 분석, 내용어(명사·수사·동사·형용사·어근·숫자·영문·한자)만 사용
- `kiwi_userdict`: `kiwi` + `configs/bm25_userdict.txt`의 회사·제품명을 고유명사로 등록
- `char2gram`: 공백을 뺀 글자 2-gram (형태소 분석 없이 한국어 어형 변화와 이름에 둔감)

토크나이저는 청크와 질문에 같은 것을 쓴다. 평가 정답·근거 원문은 이 모듈에 들어오지 않는다.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Callable

USERDICT_PATH = Path(__file__).resolve().parents[2] / "configs" / "bm25_userdict.txt"
TOKENIZER_NAMES = ("whitespace", "kiwi", "kiwi_userdict", "char2gram")

# kiwipiepy 품사 태그: 내용어만 남긴다(조사 J*, 어미 E*, 접사 X*(XR 제외), 부호 S*(SN·SL·SH 제외), 보조용언 VX, 지정사 VC* 제외)
_CONTENT_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XR", "SN", "SL", "SH", "MAG"}
_WORD_RE = re.compile(r"[0-9A-Za-z가-힣一-鿿]+")


def whitespace_tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def char_bigrams(text: str) -> list[str]:
    out: list[str] = []
    for word in whitespace_tokens(text):
        out += [word[i:i + 2] for i in range(len(word) - 1)] if len(word) > 1 else [word]
    return out


def load_userdict(path: Path = USERDICT_PATH) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


@functools.lru_cache(maxsize=2)
def _kiwi(with_userdict: bool):
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    if with_userdict:
        for word in load_userdict():
            kiwi.add_user_word(word, "NNP")
    return kiwi


def kiwi_tokens(text: str, with_userdict: bool = False) -> list[str]:
    return [t.form.lower() for t in _kiwi(with_userdict).tokenize(text) if t.tag in _CONTENT_TAGS]


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    if name == "whitespace":
        return whitespace_tokens
    if name == "kiwi":
        return lambda text: kiwi_tokens(text, False)
    if name == "kiwi_userdict":
        return lambda text: kiwi_tokens(text, True)
    if name == "char2gram":
        return char_bigrams
    raise ValueError(f"알 수 없는 토크나이저: {name!r} (가능: {', '.join(TOKENIZER_NAMES)})")
