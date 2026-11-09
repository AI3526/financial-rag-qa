# Financial RAG QA

금융 PDF 문서에서 근거를 찾아 답하는 RAG 질의응답 서비스. Naive RAG(Dense), Hybrid(Dense + BM25), Hybrid + Reranking 세 방식을 같은 조건에서 비교하고, FastAPI · LangChain · Langfuse로 연결한다.

> 작성 중입니다. 최종 선택 근거, 한계, 팀원별 기여, 실행 결과는 STEP 15에서 정리합니다.
> 설계 문서는 [project_docs/](project_docs/), 승인된 결정은 [DECISIONS.md](DECISIONS.md), 진행 상황은 [TASKS.md](TASKS.md)를 참고하세요.

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

`POST /rag/query` 본문은 `{"question": "...", "rag_method": "dense", "question_id": "선택"}`이고, 응답은 `{request_id, status, answer, reason, citations}`이다. `rag_method`는 현재 `dense`만 지원한다(`hybrid`, `hybrid_rerank`는 STEP 5~6에서 추가, 그 전에는 422).

- 오류는 모두 `status: "error"`와 `request_id`를 가진 같은 형식이다: 422 `invalid_request`, 500 `retrieval_error`·`internal_error`, 502 `model_error`·`output_validation_error`, 504 `model_timeout`.
- **STEP 8 전에는 생성 모델이 연결되지 않아** 정상 질문도 500 `internal_error`를 반환한다(답을 만들어 내지 않는다).
- 선택 헤더: `traceparent`(평가 실험 Trace에 이어 붙기), `X-Experiment-Id`, `X-Track`, `X-Run-Type`. 응답 헤더 `X-Request-Id`, `X-Trace-Id`.
- 요청마다 run 레코드가 `results/{experiment_id}/runs.jsonl`(헤더가 없으면 `results/service/runs.jsonl`, Git 제외)에 덧붙여 저장된다.
- Windows 쉘에서 curl로 한글을 보내면 UTF-8이 아닐 수 있다. 파이썬(httpx)이나 UTF-8 파일(`--data-binary @file`)로 보낸다.

## 사용 자료

`data/finance_basic/`에 제공된 합성 데이터(PDF 15개, 평가 JSON)를 사용한다. 검색 대상은 `documents/*.pdf`뿐이며 `evaluation/`의 정답과 채점 기준은 색인하지 않는다.
