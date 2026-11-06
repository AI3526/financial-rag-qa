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

## 사용 자료

`data/finance_basic/`에 제공된 합성 데이터(PDF 15개, 평가 JSON)를 사용한다. 검색 대상은 `documents/*.pdf`뿐이며 `evaluation/`의 정답과 채점 기준은 색인하지 않는다.
