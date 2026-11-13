# Financial RAG QA

금융 PDF 문서에서 근거를 찾아 답하는 RAG 질의응답 서비스. Naive RAG(Dense), Hybrid(Dense + BM25), Hybrid + Reranking 세 방식을 같은 조건에서 비교하고, FastAPI · LangChain · Langfuse로 연결한다.

> 최종 결과는 [RESULTS.md](RESULTS.md), 설계 문서는 [project_docs/](project_docs/), 승인된 결정은 [DECISIONS.md](DECISIONS.md), 진행 상황은 [TASKS.md](TASKS.md)를 참고하세요.

## 결과 요약과 최종 선택

세 방식을 같은 질문 60개(development 30 + additional 30), 같은 설정(최종 청크 5개, 문맥 4,000토큰, 생성 `gpt-6-luna`)으로 한 번씩 실행했다(답변 180개). **최종 선택은 Hybrid+Rerank**이다.

주결과(additional 30문항)

| 지표 | dense | hybrid | hybrid_rerank |
|---|---|---|---|
| Hit@5 (문서 단위) | 0.679 | 0.857 | **0.929** |
| Recall@5 | 0.536 | 0.696 | **0.786** |
| Faithfulness | **0.984** | 0.951 | 0.928 |
| Judge 사실 | 0.600 | 0.700 | **0.800** |
| Judge 조건 | 0.867 | 0.900 | 0.900 |
| Judge 근거 | 0.933 | 0.931 | **0.967** |
| Judge 판단 | 0.867 | 0.867 | 0.833 |

운영(일반 평가 180건): 세 방식 모두 시스템 오류 0건, 평균 응답시간 2.5~3.0초(rerank가 약 0.35~0.43초 느림), 60건당 비용 약 0.0175~0.0177 USD. 보안(질문 2개 × 기본/방어 프롬프트 = 답변 4개)은 공격 성공 0건이지만 공격이 부분 노출된 1건에서도 기본 프롬프트가 이미 따르지 않아 방어율은 계산할 수 없다.

**선택 이유**: 사전에 고정한 선택 순서(Judge 사실·조건·판단 → Faithfulness와 근거 → 검색 지표)에서 조건·판단은 세 방식이 ±0.05 이내로 비슷하고 **사실은 hybrid_rerank가 hybrid보다 +0.10(좋아진 3 · 나빠진 0), dense보다 +0.20(6 대 0)** 앞섰다. Faithfulness는 dense가 가장 높지만 우선순위가 더 높은 항목을 따랐다. 사람이 확인한 중대 오류는 방식 간 최대 1건 차이라 제외되는 방식이 없다(dense 1, hybrid 0, hybrid_rerank 1). 자세한 표와 근거는 [RESULTS.md](RESULTS.md), 결정 기록은 `DECISIONS.md` D-046·D-049·D-050.

## 한계

- 평가 질문이 60개(세트당 30개)라 한 문제가 3.3%p이고, 사실 항목 차이의 신뢰구간이 0에 걸린다. 생성 결과는 같은 질문도 실행마다 달라질 수 있다(1회 실행).
- 질문과 정답이 공개돼 있어 결과를 비공개 테스트 성능으로 읽으면 안 된다. 평가 모델이 생성 모델과 같다(`gpt-6-luna`, 자기 평가 편향 가능성).
- 대표 사례 18건(`results/official-001-critical/worksheet_cases.md`)의 사람 확인은 하지 못했다. 중대 오류는 64건 전수가 아니라 후보 2건과 표본 15건을 사람이 확인했다.
- 보안 비교는 질문 2개, 각 1회라 방어 효과를 평가하지 못했다. 운영 지표는 한 번 실행한 결과이며 동시성 1이다.
- Basic 코퍼스(PDF 15개) 기준 선택이다. 서비스 화면, 대화 메모리, 부하 시험 같은 서비스화 항목은 하지 않았다.

## 확장 실험 요약 (별도 브랜치, 공식 결과에 영향 없음)

`extension/intermediate` 브랜치에서 과제가 제공한 Intermediate 데이터(PDF 50개, 3,750쪽, 표가 많음)로 RAG 개선을 시도했다. 상세는 그 브랜치의 `EXTENSION_INTERMEDIATE_SUMMARY.md`에 있고 Basic의 공식 선택에는 영향을 주지 않는다.

- 병목은 검색 후보가 아니라 최종 선택이었다(정답 청크가 후보 안에 있어도 5개에서 잘림). **후보 40 + 최종 청크 12개**가 가장 크게 효과를 냈다.
- 그 위에 질문 분해(회사·기간·범위·항목 단위 짧은 질의, 분기 표기는 키워드 검색에만 병기, 질의별 자리 배정)를 얹으면 사전 고정 규칙으로 한 번만 연 확인용 질문 53개에서 사실 0.824 → 0.893으로 한 번 더 좋아졌다(조건 −1질문 경계).
- 청킹 변형 5가지, 표 행 문장화, 문서 라우팅, 재검색 등은 효과가 없어 채택하지 않았다.
- 쪽 단위 검색 지표는 같은 사실이 다른 쪽에도 있으면 못 찾음으로 세어 과소평가였다.

## 개발 환경 설정

필요: [uv](https://docs.astral.sh/uv/), Git. Python 3.12는 `uv`가 관리한다(Windows의 `python` 별칭은 사용하지 않는다).

```bash
# 1. 의존성 설치 (.venv 생성, uv.lock 기준)
uv sync

# 2. 환경 변수 설정: .env.example을 복사해 값을 채운다 (.env는 Git에 올리지 않는다)
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

`.env` 변수

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | OpenAI API 키 (비밀) |
| `OPENAI_MODEL` | 생성 모델 이름 (`gpt-6-luna`) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Langfuse 프로젝트 키 (비밀) |
| `LANGFUSE_BASE_URL` | Langfuse 호스트 (Cloud: `https://cloud.langfuse.com`) |
| `QDRANT_PATH` | Qdrant 로컬 저장소 경로 (STEP 3부터 사용) |
| `HF_HOME` | 모델 다운로드 캐시 경로. 저장소 밖을 권장 (STEP 3부터 사용) |

```bash
# 3. 연결 확인 (OpenAI 호출 몇 번과 Langfuse 테스트 Trace 1건이 실제로 전송된다)
uv run python scripts/check_connections.py all

# 4. 단위 테스트
uv run pytest
```

## PDF 추출 (STEP 1)

```bash
# documents/*.pdf 전체를 추출해 data/processed/documents.jsonl, pages.jsonl에 저장한다
# (이미 결과가 있으면 중단한다. 덮어쓰려면 --force)
uv run python scripts/ingest_pdfs.py
```

`documents.jsonl`에는 PDF별 페이지 수, sha256, 추출 상태·시간·오류를, `pages.jsonl`에는 페이지별 추출 텍스트(가공하지 않은 원문)를 저장한다. 처리에 실패한 문서나 페이지는 중단하지 않고 상태와 사유를 기록한다.

- 실행 결과는 Langfuse에도 기록된다(Trace `document-ingestion`, PDF마다 Span `extract-pdf FB01.pdf`처럼 파일명 포함, 페이지 텍스트 포함, environment `development`). 오프라인으로 실행하려면 `--no-langfuse`를 붙인다.
- 결과 파일이 이미 있으면 덮어쓰지 않고 중단한다(`--force`로 재생성).

## 청킹 (STEP 2)

```bash
# pages.jsonl을 청킹해 data/processed/chunks.jsonl, chunking_manifest.json에 저장한다
uv run python scripts/chunk_pages.py
```

페이지 경계를 넘지 않고, 문단을 450자까지 묶는다(450자 초과 문단은 800자 이내면 통째로 유지). 머리말·문서 제목·소제목은 `metadata`로 옮기고 꼬리말은 버린다. 설정과 이유는 `DECISIONS.md` D-025. Trace `document-chunking`(문서마다 Span `chunk-pdf FB01.pdf`, 청크 텍스트 포함)으로 Langfuse에도 기록되며 `--no-langfuse`로 끌 수 있다. 결과가 있으면 `--force` 없이는 중단한다.

## 임베딩과 색인 (STEP 3)

`.env`에 `QDRANT_URL`, `QDRANT_API_KEY`를 넣으면 Qdrant Cloud에, 비워 두면 `QDRANT_PATH`(기본 `data/qdrant`)의 로컬 폴더에 저장한다.

```bash
# 컬렉션마다 한 번씩 (이미 있으면 덮어쓰지 않고 중단한다)
uv run python scripts/build_index.py --model e5
uv run python scripts/build_index.py --model bgem3
# development 30문항으로 두 후보를 비교한다 (결과는 results/dev-embed-select-001/)
uv run python scripts/compare_embeddings.py
```

- 공식 임베딩 모델은 `intfloat/multilingual-e5-large`로 동결했다(`DECISIONS.md` D-026, D-027). BGE-M3 컬렉션과 비교 결과는 확장 실험을 위해 보존한다.
- 모델은 처음 실행할 때 내려받아 HF 캐시(`C:\Users\<사용자>\.cache\huggingface\hub`, 저장소 밖)에 저장된다. `.env`의 `HF_HOME`을 비워 두면 기본 위치를 쓴다(빈 값은 설정하지 않은 것으로 처리). GPU는 CUDA 12.8 이상용 PyTorch가 필요하다.
- Langfuse에는 색인(`document-indexing`)과 비교(`embedding-comparison`)가 기록된다. `--no-langfuse`로 끌 수 있다(색인 스크립트).

## Dense 검색 (STEP 4)

```python
from src.config import load_env
load_env()  # 다른 라이브러리보다 먼저
from src.retrieval.dense import DenseRetriever

result = DenseRetriever().retrieve("루미센서 E는 어떤 제품인가요?", k=5)
for c in result.chunks:   # rank, score, chunk_id, book_id, file_name, page, text, metadata
    print(c["rank"], c["score"], c["file_name"], c["page"])
```

`DenseRetriever(langfuse=get_langfuse())`처럼 Langfuse를 주면 검색마다 Span `dense-retrieve`가 기록된다. 질문 유형별 동작 확인은 `uv run python scripts/check_dense_retrieval.py`로 실행하며 결과는 `results/dense-check-001/`에 저장되고 Langfuse에도 올라간다.

동결된 e5-large와 Qdrant 컬렉션 `basic__chunk-v1__e5large-v1`을 쓴다. 빈 질문은 `InvalidQuery`, 저장소 오류는 `RetrievalError`를 낸다. 유사도 임계값은 두지 않으며(D-028) 검색은 항상 Top-k를 돌려준다.

## API 서버 (STEP 7)

```bash
uv run uvicorn --factory src.api.main:build_default_app --port 8000
curl http://127.0.0.1:8000/health
```

`POST /rag/query` 본문은 `{"question": "...", "rag_method": "dense", "question_id": "선택"}`이고, 응답은 `{request_id, status, answer, reason, citations}`이다. `rag_method`는 `dense`, `hybrid`, `hybrid_rerank`를 지원한다. `hybrid_rerank`는 `hybrid` 상위 20개 후보를 `BAAI/bge-reranker-v2-m3`(처음 실행할 때 약 2.3GB를 내려받고 GPU가 있으면 사용)로 재정렬해 최종 5개를 쓴다. `hybrid`는 Dense 20 + BM25 20 후보를 RRF(k=60)로 결합해 최종 5개를 쓴다(BM25 토크나이저는 공백 분리로 동결, D-035). 검색 방식 비교는 `uv run python scripts/compare_retrieval.py`(결과는 `results/`에 로컬 저장, 폴더가 있으면 중단).

- 오류는 모두 `status: "error"`와 `request_id`를 가진 같은 형식이다: 422 `invalid_request`, 500 `retrieval_error`·`internal_error`, 502 `model_error`·`output_validation_error`, 504 `model_timeout`.
- `.env`에 `OPENAI_API_KEY`와 `OPENAI_MODEL`이 있으면 gpt-6-luna 생성기(LangChain)로 답하고, 없으면 생성 모델이 연결되지 않아 정상 질문도 500 `internal_error`를 반환한다(답을 만들어 내지 않는다).
- 프롬프트는 `prompts/answer_v2.json`(기본, Git 원본, sha256과 버전을 run 레코드에 기록. 바꿀 때는 새 버전 파일을 추가하고 기존 파일은 덮어쓰지 않는다)이고, 최종 문맥은 토큰 한도(추정 4,000) 안에서 낮은 순위 청크를 통째로 제외한다. 생성 설정(`reasoning_effort=low`, 최대 출력 토큰 1,500, timeout 30초, 최대 모델 호출 3회 등)은 공식 실행 전에 동결했다(`configs/official_freeze.json`, D-045).
- 프롬프트 점검(로컬 전용): `uv run python scripts/run_dev_questions.py --prompt answer_v2 --experiment-id <새 ID> [--rag-method dense|hybrid|hybrid_rerank]`(development 30문항의 기대 행동 비교), `scripts/run_exception_questions.py`(예외 질문 6종). 결과 폴더가 있으면 덮어쓰지 않고 중단한다.
- 모델 단가는 `configs/pricing.json`(출처와 확인 날짜 포함)이고, run 레코드에 요청별 `cost`가 기록된다. 캐시 입력 토큰은 단가가 달라 `llm_calls`에 따로 기록한다(D-033).
- 선택 헤더: `traceparent`(평가 실험 Trace에 이어 붙기), `X-Experiment-Id`, `X-Track`, `X-Run-Type`. 응답 헤더 `X-Request-Id`, `X-Trace-Id`.
- 요청마다 run 레코드가 `results/{experiment_id}/runs.jsonl`(헤더가 없으면 `results/service/runs.jsonl`, Git 제외)에 덧붙여 저장된다.
- Windows 쉘에서 curl로 한글을 보내면 UTF-8이 아닐 수 있다. 파이썬(httpx)이나 UTF-8 파일(`--data-binary @file`)로 보낸다.

## 답변 평가: RAGAS Faithfulness (STEP 11)

```bash
# ragas는 메인 환경의 openai 버전을 바꾸므로 격리 환경에서만 실행한다(결과 폴더가 있으면 중단)
uv run --with ragas --with "langchain-community<0.4" python scripts/run_ragas.py \
    --source <서비스 실험 폴더> --out <새 결과 폴더> [--passes 3] [--langfuse]
# ragas 환경의 테스트
uv run --with ragas --with "langchain-community<0.4" pytest tests/ragas
```

입력은 서비스가 남긴 `results/<source>/runs.jsonl`(질문, 답변, **실제 최종 문맥**)이고 정답 파일은 읽지 않는다. 사실 주장이 없는 답변은 N/A, 평가 호출 실패는 평가 불가로 기록하며 0점으로 바꾸지 않는다. 주장별 판정 이유를 저장한다. development는 1회, 공식 실행은 `--passes 3`(평균과 범위 기록)이다. ragas의 사용 통계 전송은 끈다(`RAGAS_DO_NOT_TRACK=true`). 설정과 한계는 `DECISIONS.md` D-039.

## 답변 평가: LLM-as-a-Judge (STEP 11)

```bash
# 사실·조건·근거·판단 4항목 채점 (결과 폴더가 있으면 중단)
uv run python scripts/run_judge.py --source <서비스 실험 폴더> --out <새 결과 폴더> [--prompt judge_v1|judge_v2] [--passes N] [--langfuse]
```

입력은 서비스가 남긴 `results/<source>/runs.jsonl`과 평가 파일의 참고 정답·필수 사실·금지 주장·기대 동작·채점 설명이다(`evidence_groups`는 넣지 않음). 항목마다 1 / 0 / 해당 없음 / 평가 불가와 짧은 이유를 남기고 평가 불가는 코드가 정한다. 서비스 상태가 기대 동작과 다르면 판단은 코드가 0으로 기록하며, 서비스 오류 응답은 모델을 부르지 않고 판단 0·나머지 평가 불가로 기록한다. 프롬프트는 `prompts/judge_v2.json`(기본, 사용자 승인 문안: 근거 항목의 부정 진술 규칙, 쪽 번호 출처 표, 필수 사실이 최종 문맥에 있었는지의 진단 추가)이며 `--prompt judge_v1`로 이전 문안을 쓸 수 있다. `--passes 3`은 같은 답변을 3회 독립 채점해 요청별 다수결(`judge_majority.jsonl`)과 회차별 충족률·범위·판정이 갈린 답변 수를 `summary.json`에 함께 남긴다(공식 실행용, 개발은 기본 1회). 설정과 한계는 `DECISIONS.md` D-040, D-043, D-044, D-045.

## 검색 평가: Hit@k·Recall@k (STEP 11)

```bash
# 서비스 실행 기록(runs.jsonl)의 검색 결과로 계산한다. 모델 호출 없음, 결과 폴더가 있으면 중단
uv run python scripts/evaluate_retrieval.py --source <실험 폴더> [<실험 폴더> ...] --out <새 결과 폴더>
```

k는 답변 생성 직전에 선택한 상위 청크 수이며 문서(`book_id`) 단위로 센다. 참고 근거가 비면 해당 없음, 서비스 오류로 검색 기록이 없으면 평가 불가, 검색 결과가 비어도 참고 문서가 있으면 0으로 계산한다. 문맥 토큰 한도 이후의 최종 문맥 기준 지표와 (문서, 쪽) 단위 참고 지표도 함께 낸다. 사람이 대표 사례를 원문과 대조하는 작업지는 `results/human-check-001/worksheet.md`다. 설정과 결과는 `DECISIONS.md` D-042.

## 공식 비교 실행과 평가 (STEP 12)

질문 60개 × 세 방식 = 답변 180개를 실제 서버에 HTTP로 한 번에 하나씩(동시성 1) 보낸다. 질문 파일(`*_questions.json`)에는 번호와 질문뿐이고 정답은 서비스에 보내지 않는다. 공식 실행(`official-001`)은 Git 작업 트리가 clean일 때만 허용되고, 동결 파일(`configs/official_freeze.json`)과 현재 상태가 한 글자라도 다르면 중단한다. 결과 폴더가 이미 있으면 모든 스크립트가 덮어쓰지 않고 중단한다.

```bash
# 1. 서버를 띄운다 (별도 터미널)
uv run uvicorn --factory src.api.main:build_default_app --port 8000
# 2. 사전 점검과 manifest 작성 (clean 작업 트리 필요), 3. 공식 실행
uv run python scripts/preflight_official.py --write-manifest official-001
uv run python scripts/run_official.py --experiment-id official-001
# 4. 평가: 검색 지표(모델 호출 없음), Judge 3회, Faithfulness 3회(ragas는 격리 환경)
uv run python scripts/evaluate_retrieval.py --source official-001 --out official-001-retrieval
uv run python scripts/run_judge.py --source official-001 --out official-001-judge --passes 3 --langfuse
uv run --with ragas --with "langchain-community<0.4" python scripts/run_ragas.py \
    --source official-001 --out official-001-faithfulness --passes 3 --langfuse
# 5. 비교표 (지표를 하나로 합산하지 않고 additional을 주결과로 낸다)
uv run python scripts/compare_official.py --source official-001 --retrieval official-001-retrieval \
    --faithfulness official-001-faithfulness --judge official-001-judge --out official-001-compare
# 6. 중대 오류 분류 입력과 사람 확인 작업지
uv run python scripts/make_cases_official.py export --source official-001 --judge official-001-judge --out official-001-critical
uv run python scripts/make_cases_official.py worksheet --source official-001 --compare official-001-compare --out official-001-critical
# 7. (선택) 최종 평가 값을 Langfuse Score로 올린다: 기본은 계획만 출력, --apply가 실제 기록
uv run python scripts/post_scores.py --compare official-001-compare
```

- 소량 점검은 `run_official.py --track development --allow-dirty --limit 2 --dataset dryrun-basic-v1`로 한다.
- 중대 오류 분류(`critical_classification.jsonl`)는 이 스크립트가 만들지 않는다. 입력을 읽은 Claude Code 세션이 분류를 직접 작성했고, 후보와 표본을 사람이 원문과 대조해 `worksheet_critical.md`에 기록했다(요약 `human_review_summary.json`).
- 선택 규칙은 `project_docs/01_core/12_ACCEPTANCE_CRITERIA.md` §4.3이며 비교 스크립트는 최종 방식을 정하지 않는다. 설정과 결과는 `DECISIONS.md` D-045·D-046·D-049.

## 보안 비교 (STEP 13)

선택한 구성(hybrid_rerank) 하나로 보안 질문 2개(FPAIR01, FPAIR02)를 기본 프롬프트(`answer_v2`)와 방어 프롬프트(`answer_v2_defense`)로 실행한다(답변 4개). 질문마다 한 번만 검색하고 같은 최종 문맥을 두 프롬프트에 그대로 넣는다(재검색 없음).

```bash
uv run python scripts/run_security.py --experiment-id security-001      # clean 작업 트리 필요
uv run python scripts/verify_security_langfuse.py security-001          # Langfuse 기록 대조(읽기 전용)
```

실행 전에 프롬프트 해시와 공격 노출 판정 규칙(저장된 최종 문맥에 공격 문장이 완전·부분(연속 20자 이상)·미노출 중 어느 수준으로 들어갔는지)을 `rules_frozen.json`에 고정한다. 분석은 `results/security-001/analysis.md`, 결정은 `DECISIONS.md` D-050이다. 소량 점검은 `--experiment-id security-smoke-001 --track development --no-langfuse --allow-dirty --limit 1`이다.

## 운영 지표 (STEP 14)

```bash
uv run python scripts/ops_summary_official.py          # 필수 지표: 정상 건 평균 응답시간·건수, 오류율, 토큰, 비용
uv run python scripts/ops_percentiles_official.py      # [확장] P50·P90·P95·P99 (방식당 60건이라 P95·P99는 불안정)
```

읽기 전용이며 새 모델·검색 호출이 없다. 결과는 `results/official-001-ops/`에 저장된다. 보안·평가 호출은 일반 질문의 운영 지표와 구분해 따로 집계한다(`DECISIONS.md` D-048).

## 디렉터리 안내

| 위치 | 내용 |
|---|---|
| `src/` | 추출·청킹(`ingest`), 임베딩·색인(`index`), 검색(`retrieval`), 생성(`generation`), API(`api`), 평가(`evaluation`), 관측(`observability`) |
| `scripts/` | 단계별 실행 스크립트 |
| `prompts/` | 프롬프트 원본(답변 `answer_v2`, 방어 `answer_v2_defense`, Judge `judge_v2` 등. 바꿀 때는 새 버전 파일을 추가한다) |
| `configs/` | 동결 설정(`official_freeze.json`), 모델 단가(`pricing.json`) |
| `results/` | 실험 결과(`official-001*`, `security-001` 등). 기존 결과는 덮어쓰지 않는다 |
| `project_docs/` | 과제 원본(`00_assignment`)과 설계 문서 |
| `RESULTS.md`, `DECISIONS.md`, `TASKS.md` | 최종 결과, 승인된 결정, 작업 상태 |

## 사용 자료

`data/finance_basic/`에 제공된 합성 데이터(PDF 15개, 평가 JSON)를 사용한다. 검색 대상은 `documents/*.pdf`뿐이며 `evaluation/`의 정답과 채점 기준은 색인하지 않는다.
