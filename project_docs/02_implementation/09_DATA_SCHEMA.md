# 09. DATA_SCHEMA.md
# 데이터 구조 및 API 계약

## 1. 문서 목적

이 문서는 Financial RAG QA 프로젝트에서 사용하는 주요 데이터의 구조, 식별자, 관계와 API 계약을 정의한다.

- 필드명과 자료형을 일관되게 유지한다.
- 서로 다른 RAG 방식의 결과를 동일한 구조로 저장하고 비교한다.
- 질문, 검색 결과, 실제 모델 입력, 답변, 평가 결과를 연결한다.
- Langfuse Trace와 프로젝트 내부 데이터의 연결 관계를 명확히 한다.
- 평가 정답 및 채점 기준이 검색이나 답변 생성에 유입되지 않도록 분리한다.

데이터의 **논리적 의미와 연결 관계**를 정의하며, 저장은 평탄화한다(§3). 필수 구현은 Basic 단일 데이터셋 기준이다.

### 1.1 설계 원칙

**[필수]**
- 필드명과 자료형을 명확하게 정의하고, 데이터 간 연결을 위해 고유 식별자를 사용한다.
- 동일한 평가 질문을 여러 RAG 방식에서 공통으로 사용한다.
- 검색 결과와 최종 모델 입력 문맥을 구분한다.
- 실제 모델 입력·출력과 평가 데이터를 분리한다.
- 데이터가 없거나 확인할 수 없는 경우와 값이 0인 경우를 구분한다.
- 오류 및 실패한 실행도 기록한다.
- 없는 메타데이터를 임의로 만들지 않는다.

**[확장]**
- 관계형 데이터베이스(SQLite, PostgreSQL), 데이터 웨어하우스, 스키마 자동 검증·마이그레이션
- Observation 단위의 별도 저장, 사용자별 권한·접근 이력, 캐시 데이터

## 2. 용어 통일

| 용어 | 의미 |
|---|---|
| `book_id` | PDF 파일명에서 `.pdf`를 뺀 값(예: `FB01`). 문서 식별자이며 `document_id`는 사용하지 않는다. 코드에서 `file_id`를 쓰면 같은 값이다. |
| `file_name` | PDF 파일명(예: `FB01.pdf`). 변경하지 않는다. |
| `page` | PDF의 실제 페이지 번호. 첫 페이지가 1이다. |
| `question_id` | 평가 JSON의 `id`. 동일 질문을 여러 RAG 방식으로 실행해도 유지한다. |
| `track` | 실험 단위 구분: `official` / `extension` / `development` |
| `run_type` | 요청 단위 구분: `general_eval` / `security` / `mock_error` / `scoring` |

## 3. 저장 구조

저장은 파일 기반 JSON/JSONL이다(관계형 DB는 [확장]). 요청 1건의 검색·문맥·답변·사용량·성능·오류는 **run 레코드 1개**에 중첩 객체로 담는다.

| 경로 | 내용 |
|---|---|
| `data/finance_basic/documents/*.pdf`, `evaluation/*.json` | 제공 원본. 수정하지 않는다. |
| `data/processed/documents.jsonl` | PDF별 처리 매니페스트(§6.1) |
| `data/processed/pages.jsonl` | 페이지별 추출 결과(§6.2) |
| `data/processed/chunks.jsonl` | 청크(§6.3) |
| `data/processed/chunking_manifest.json` | 청킹 설정(`params`), 통계(청크 수·평균/최소/최대 글자 수), 경고 |
| `data/processed/index_manifest__{embedding_version}.json` | 컬렉션(임베딩 모델)별 색인 정보와 처리 통계(§7). 예: `index_manifest__e5large-v1.json` |
| `results/{experiment_id}/manifest.json` | 실행 설정 스냅샷(§12) |
| `results/{experiment_id}/runs.jsonl` | 요청별 run 레코드(§9) |
| `results/{experiment_id}/evaluations.jsonl` | 질문×방식별 평가 결과(§11) |
| `results/{experiment_id}/security.jsonl` | 보안 비교 실행 판정(§13) |
| `results/{experiment_id}/summary.json` | 집계 결과 |

벡터는 Qdrant가 보관하며 별도 JSON으로 중복 저장하지 않는다. 결과 JSON/JSONL이 비교표의 기준 데이터이고 Langfuse는 추적용이다(`05_OBSERVABILITY_PLAN.md`).

## 4. 식별자와 공통 규칙

### 4.1 식별자 6종

| 필드 | 자료형 | 설명 |
|---|---|---|
| `request_id` | string | API 요청 1건의 ID. 서버가 UUID로 생성한다. |
| `trace_id` | string | Langfuse Trace ID. `request_id`에서 결정적으로 만들고 둘 다 저장한다. 생성은 `Langfuse.create_trace_id(seed=request_id)`이며 같은 seed는 같은 값을 만든다(STEP 0에서 langfuse 5.2.0으로 확인). |
| `experiment_id` | string | 실험 실행 또는 비교 세트의 ID |
| `question_id` | string/null | 평가 질문 ID. 평가 실행에서만 값이 있다. |
| `chunk_id` | string | 청크 고유 ID. 형식 예: `FB01-p1-c1` |
| `book_id` | string | 문서 ID |

식별자 역할은 서로 섞지 않는다.
- 하나의 요청은 하나의 Trace와 연결된다. `request_id`와 `trace_id`는 서로 다른 식별자이다.
- 한 실험에 여러 요청이 포함된다.
- 같은 `question_id`를 세 RAG 방식으로 실행하면 서로 다른 `request_id`를 가진 run 3개가 생긴다. 질문×방식은 `(experiment_id, question_id, rag_method)`로 식별한다.
- 재시도는 별도 요청이 아니라 같은 `request_id`의 시도이다.

### 4.2 값 규칙

- 시간은 UTC datetime(ISO 8601), 실행 시간은 ms 단위이다. 토큰 수는 정수이다.
- 확인할 수 없는 토큰 수나 비용은 `null`이며 0으로 저장하지 않는다.
- 항목별 평가 상태: `evaluated`(정상 평가) / `not_applicable`(해당 없음) / `not_evaluated`(평가 불가) / `error`(평가 실행·계산 오류). 점수 필드는 `evaluated`일 때만 0 또는 1(또는 float)이고 나머지는 `null`이다.
- 사용량 상태: `confirmed` / `partial` / `unknown`

## 5. 평가 데이터와 서비스 데이터의 분리 [필수]

다음 평가 전용 정보는 PDF 추출, 청킹, 임베딩, 검색, 최종 문맥 구성, 답변 생성 입력에 들어가면 안 된다.

- `reference_answer`, `required_facts`, `forbidden_claims`, `evidence_groups`, `evidence_required`, `grading_notes`
- `expected_action`
- 보안 사례의 `attack_text` 등 공격 라벨 필드, `bad_claims`

API 요청에는 `question`, `rag_method`, 선택 `question_id`만 있다. `question_id`는 기록·연결용이며 검색과 생성에 사용하지 않는다. run 레코드(§9)에는 평가 전용 정보를 넣지 않고, 평가 단계(§11)에서만 답변 생성 후 불러온다.

## 6. 원본 문서·추출·청크

### 6.1 Document 매니페스트 (`documents.jsonl`)

PDF 1개당 1행이다.

| 필드 | 자료형 | 필수 | 설명 |
|---|---|---|---|
| `book_id` | string | 필수 | 파일명에서 `.pdf`를 뺀 값 |
| `file_name` | string | 필수 | 원본 파일명 |
| `file_path` | string | 선택 | 원본 위치 |
| `file_sha256` | string | 필수 | 원본 파일의 sha256 |
| `file_size` | integer | 필수 | 바이트 |
| `page_count` | integer/null | 필수 | PDF 페이지 수 |
| `dataset_id`, `dataset_version` | string | 필수 | 예: `basic`, `basic-v1` |
| `extraction_status` | string | 필수 | `success` / `partial` / `failed` |
| `extraction_seconds` | float/null | 필수 | 추출 시간 |
| `error_message` | string/null | 선택 | 추출 오류 |
| `created_at` | datetime | 필수 | 처리 시간 |

문서 제목, 기업명 등은 원문을 확인한 값만 선택 필드로 추가할 수 있으며 Basic에서는 채우지 않는다. 파일명만으로 추정해 확정하지 않는다.

### 6.2 Extracted Page (`pages.jsonl`)

| 필드 | 자료형 | 필수 | 설명 |
|---|---|---|---|
| `book_id` | string | 필수 | 문서 ID |
| `page` | integer | 필수 | PDF 실제 페이지 번호 |
| `text` | string | 필수 | 추출 텍스트 |
| `extraction_method` | string | 필수 | 추출 라이브러리/방법 |
| `extraction_version` | string | 필수 | 추출기·설정 버전 |
| `extraction_status` | string | 필수 | `success` / `partial` / `failed` |
| `error_message` | string/null | 선택 | 오류 내용 |
| `extracted_at` | datetime | 필수 | 추출 시간 |

표, 숫자, 단위, 각주가 누락·훼손될 수 있으므로 대표 PDF를 원본과 대조한다.

### 6.3 Chunk (`chunks.jsonl`)

| 필드 | 자료형 | 필수 | 설명 |
|---|---|---|---|
| `chunk_id` | string | 필수 | 청크 고유 ID |
| `book_id` | string | 필수 | 소속 문서 |
| `file_name` | string | 필수 | 원본 파일명 |
| `page` | integer | 필수 | 청크가 속한 페이지 (단일) |
| `chunk_text` | string | 필수 | 청크 본문 |
| `chunk_index` | integer | 필수 | 문서 내 청크 순서 |
| `chunking_method`, `chunking_version` | string | 필수 | 청킹 방식·설정 버전 |
| `dataset_id`, `dataset_version` | string | 필수 | 데이터셋 정보 |
| `token_count` | integer/null | 선택 | 청크 토큰 수 |
| `metadata` | object | 선택 | 원문에서 확인한 메타데이터만. chunk-v1은 페이지의 `header`(머리말), `doc_title`, `section_title`을 둔다(구조가 다른 페이지는 빈 객체). |
| `created_at` | datetime | 필수 | 생성 시간 |

규칙:
- **청크는 페이지 경계를 넘지 않고 단일 `page`를 가진다.** 여러 페이지에 걸친 청크(`page_spans: [{page, char_start, char_end}]`)는 [확장]이다. 합쳐진 내용을 한 페이지인 것처럼 표시하지 않는다.
- 표의 값은 행·열 제목, 단위, 각주와 함께 보존한다.
- `metadata`에는 원문에서 확인한 값만 쓴다. 회사·사업부문·제품·지역 등의 보강은 [확장]이다.
- `chunk_text`는 본문만 담고(머리말·꼬리말 제외), 임베딩·BM25에 넣는 검색용 텍스트는 `metadata`의 머리말·제목·소제목을 접두로 붙여 코드(`retrieval_text`)에서 만든다. 저장하지 않는다.
- 청킹 설정이 바뀌면 기존 청크를 덮어쓰지 않고 새 `chunking_version`으로 만든다.

## 7. 색인 매니페스트 (`index_manifest__{embedding_version}.json`)

컬렉션마다 파일 하나를 만든다(후보 비교 때문에 컬렉션이 둘 이상이므로). 아래 필드 외에 구현은 `embedding_license`, `max_seq_length`, 접두어, `device`, `qdrant_mode`, `token_stats`, PDF별 `per_pdf`(페이지 수·추출 상태·추출 시간·청크 수·임베딩 시간), `verification`(저장 수, 탐침 검색), `status`를 함께 기록한다.

| 필드 | 자료형 | 설명 |
|---|---|---|
| `collection_name` | string | 예: `basic__chunk-v1__e5large-v1` |
| `embedding_model`, `embedding_version` | string | 모델 ID와 확인된 revision |
| `embedding_dimension` | integer | 벡터 차원 |
| `distance`, `normalized` | string, boolean | 거리 계산 방식, 정규화 여부 |
| `chunking_version` | string | 사용한 청킹 버전 |
| `document_count`, `chunk_count`, `failed_chunk_count` | integer | 색인 통계 |
| `avg_chunk_size`, `chunk_size_unit` | float, string | 평균 청크 크기와 단위(토큰/문자) |
| `indexing_seconds` | object/float | 단계별(추출·청킹·임베딩·저장)·전체 색인 시간 |
| `created_at` | datetime | 색인 시간 |

임베딩 모델이 바뀌면 새 컬렉션을 만들어 검증하고 기존 컬렉션은 삭제하지 않는다. 일부 청크만 실패한 경우 전체 색인 성공으로 처리하지 않는다.

## 8. API 계약

### 8.1 엔드포인트

| API | 설명 |
|---|---|
| `GET /health` | 서비스가 응답하는지 확인 (단일 엔드포인트) |
| `POST /rag/query` | 질문에 대한 RAG 실행 |

### 8.2 요청

```json
{
  "question": "해온솔루션의 상반기 매출과 영업이익이 모두 증가했는가?",
  "rag_method": "hybrid",
  "question_id": "F-DEV-01"
}
```

| 필드 | 자료형 | 필수 | 설명 |
|---|---|---|---|
| `question` | string | 필수 | 사용자 질문. 공백만 있거나 빈 문자열이면 422 |
| `rag_method` | string | 필수 | `dense` / `hybrid` / `hybrid_rerank`. 그 외 값은 422 |
| `question_id` | string | 선택 | 평가 실행의 기록·연결용 |

### 8.3 응답

```json
{
  "request_id": "7f6c0e1a-…",
  "status": "answered",
  "answer": "2026년 상반기 ...",
  "reason": "정정 공지가 기존 값을 바꾸고 질문의 기준일에 적용됩니다.",
  "citations": [
    {"source_id": "S2", "book_id": "FB05", "file_name": "FB05.pdf", "page": 1}
  ]
}
```

| 필드 | 설명 |
|---|---|
| `request_id` | 서버가 생성한 요청 ID |
| `status` | `answered` / `needs_clarification` / `abstained` / `error` |
| `answer` | 답변 또는 확인할 질문 (오류 시 일반화된 안내 문구) |
| `reason` | 판단 이유. `error`일 때는 아래 오류 사유 코드 |
| `citations` | 코드가 대응표에서 변환한 출처 목록. 없으면 빈 배열 |

문서명과 페이지는 LLM이 만들지 않는다.

### 8.4 LLM 출력 계약

LLM은 아래 JSON 하나를 반환한다.

```json
{"status": "answered", "answer": "…", "reason": "…", "source_ids": ["S2"]}
```

코드에서 검증한다.
- `status`는 `answered` / `needs_clarification` / `abstained` 중 하나(`error`는 LLM이 반환하지 않는다).
- 필수 필드(`status`, `answer`, `reason`, `source_ids`)가 있어야 한다.
- `source_ids`의 모든 ID가 이번 요청의 최종 문맥(`S1`, `S2`…)에 있어야 한다.
- 위반하면 정상 답변으로 통과시키지 않고 `output_validation_error`로 처리하거나 정한 횟수 안에서 다시 생성한다(재생성의 토큰·시간도 같은 요청에 합산).
- 검색 결과가 비어 있으면 검색한 자료에서 확인하지 못했다고 답하고 `source_ids`는 빈 배열이다. "세상에 그런 자료가 없다"고 단정하지 않는다.

### 8.5 HTTP 상태와 오류 사유

| 상황 | HTTP | `status` | `reason` |
|---|---|---|---|
| 답변 / 추가 확인 / 보류 | 200 | `answered` / `needs_clarification` / `abstained` | LLM의 판단 이유 |
| 입력 검증 실패 (빈 질문, 지원하지 않는 `rag_method`) | 422 | `error` | `invalid_request` |
| 모델 호출 실패 | 502 | `error` | `model_error` |
| 모델 시간 초과·재시도 한도 도달 | 504 | `error` | `model_timeout` |
| LLM 출력 검증 실패(재생성 한도 도달) | 502 | `error` | `output_validation_error` |
| 검색 저장소 오류 | 500 | `error` | `retrieval_error` |
| 예상하지 못한 내부 오류 | 500 | `error` | `internal_error` |

- 모든 오류 응답은 위 응답 형식과 `request_id`를 포함한다.
- HTTP 성공(200)은 처리 결과이며 답변 내용의 정확성과 별개이다. `answered`는 정답 보증이 아니다.
- 추가 확인과 답변 보류는 시스템 오류가 아니다.
- **오류율 집계에는 시스템 오류(`invalid_request` 제외)만 포함한다.** 빈 질문·모의 오류·보안 비교는 일반 평가 오류율에 섞지 않는다(`run_type`).
- 비밀 값은 응답과 로그에 노출하지 않는다.

## 9. Run 레코드 (`runs.jsonl`)

요청 1건당 1행이다. 평가 전용 정보는 포함하지 않는다(§5).

| 필드 | 자료형 | 설명 |
|---|---|---|
| `request_id`, `trace_id` | string | 식별자 |
| `experiment_id`, `track` | string | 소속 실험, `official`/`extension`/`development` |
| `run_type` | string | `general_eval` / `security` / `mock_error` / `scoring` |
| `question_id` | string/null | 평가 질문 ID |
| `evaluation_set_type` | string/null | `development` / `additional` / `expansion_validation` (질문 파일 기준 구분) |
| `rag_method` | string | `dense` / `hybrid` / `hybrid_rerank` |
| `question` | string | 실제 질문 |
| `versions` | object | `config_version`, `prompt_version`, `model_id`, `dataset_version` |
| `retrieval` | object | 아래 참조 |
| `final_context` | object | 아래 참조 |
| `model_input` | string/array | 실제 모델에 전달한 프롬프트(메시지) 전체 |
| `llm_output_raw` | string/null | 모델이 반환한 원문 |
| `llm_output_parsed` | object/null | 검증을 통과한 `{status, answer, reason, source_ids}` |
| `response` | object | API가 반환한 응답(§8.3) |
| `http_status` | integer | 반환한 HTTP 상태 |
| `llm_calls` | array | 모델 호출별 사용량(아래) |
| `performance` | object | 아래 참조 |
| `error` | object/null | 오류 정보(아래) |
| `observability` | object | `langfuse_recorded`(boolean), `langfuse_error`(string/null) |
| `received_at`, `completed_at` | datetime | 요청 수신·완료 시간 |

`retrieval`:
- `dense`, `sparse`, `fused`: 각 후보 배열 `[{chunk_id, book_id, page, rank, score}]`. 사용하지 않은 방식은 `null`(0이나 빈 배열로 바꾸지 않는다). 점수는 방식마다 의미가 다르므로 단순 비교하지 않는다.
- `reranked`: `[{chunk_id, rank_before, rank_after, score}]` 또는 `null`
- `candidate_k`, `final_k`, `durations_ms`(`dense`, `sparse`, `fusion`, `rerank`)
- 검색 결과가 없어도 검색 실행 자체는 기록한다.

`final_context`: 답변 생성에 실제 전달한 문맥이며 검색 결과 전체와 구분한다.
- `context_limit_tokens`, `context_token_count`
- `chunks`: `[{source_id, chunk_id, book_id, file_name, page, rank, text}]` — 출처 ID(`S1`…)와 문서·페이지의 대응표 역할을 한다. 평가의 `retrieved_contexts`는 이 `text`를 사용한다.

`llm_calls`(재시도·재생성 시도마다 1행): `{call_index, stage, model_id, input_tokens, output_tokens, reasoning_tokens, duration_ms, retry_index, status, finish_reason, usage_status}`. `gpt-6-luna`는 추론 모델이라 `output_tokens`에 추론 토큰이 포함되므로 `reasoning_tokens`를 별도로 기록한다(STEP 0 확인). 확인 불가한 값은 `null`이다. 서비스 답변 생성만 기록하며 평가 호출은 §11에 둔다.

`performance`: `total_duration_ms`(API 수신~최종 응답, 재시도 포함), `retrieval_duration_ms`, `reranking_duration_ms`, `generation_duration_ms`, `retry_count`, `final_status`. 실패 요청도 기록한다.

`error`(`status = error`일 때): `reason`(§8.5 코드), `error_stage`, `error_message`(비밀 값 제외), `retry_count`, `is_retryable`.

비용은 실제 확인 가능한 단가가 있을 때만 별도 필드(`cost`, `currency`, `price_source`)로 계산하며, 모르면 `null`로 둔다.

## 10. 평가 데이터 (제공 원본)

### 10.1 Evaluation Question

질문 파일(`*_questions.json`)은 `id`, `question`만 가진다. 정답 파일은 다음 필드를 가지며 **원래 구조를 보존**한다.

| 필드 | 설명 |
|---|---|
| `id` | 질문 ID (`question_id`) |
| `question` | 사용자 질문 |
| `reference_answer` | 참고 정답 |
| `required_facts` | 필수 사실 목록 |
| `forbidden_claims` | 금지해야 할 주장 목록 |
| `expected_action` | `answer` / `clarify` / `abstain` |
| `evidence_groups` | `[[{book_id, page, quote}, ...], ...]` — 참고 근거 그룹의 배열 |
| `evidence_required` | 근거 필요 여부(boolean) |
| `grading_notes` | 채점 설명 |

`evaluation_set_type`은 파일 출처로 정한다: `development` / `additional` / `expansion_validation`. 이 값은 사용자 질문의 분류 라벨이 아니라 결과를 세트별로 분리 보고하기 위한 구분이다(Happy/Edge/Negative 같은 유형 라벨은 사용하지 않는다). 분석용 선택 태그는 [확장]에서 추가할 수 있다.

### 10.2 보안 사례

`security_tasks.json`(FPAIR01, FPAIR02)은 위 필드에 `attack_type`, `external_source_id`, `bad_claims`, `attack_text`, `source_question_id`, `source_page`, `integrated_document`가 추가된다. 공격 노출 판정에만 쓰이며 검색·생성 입력에 사용하지 않는다.

`expected_action`과 API 상태의 대응: `answer`→`answered`, `clarify`→`needs_clarification`, `abstain`→`abstained`. 이름만으로 채점하지 않는다.

## 11. 평가 결과 (`evaluations.jsonl`)

`(experiment_id, question_id, rag_method)`당 1행이며 해당 run의 `request_id`로 연결한다. 답변 생성이 끝난 뒤에만 생성한다.

| 필드 | 설명 |
|---|---|
| `experiment_id`, `question_id`, `rag_method`, `request_id`, `evaluation_set_type` | 키와 연결 |
| `expected_action` | 평가 시점에 읽은 기대 행동 |
| `search_eval` | 검색 평가(아래) |
| `faithfulness` | `{score, status, reason}` — RAGAS Faithfulness |
| `judge` | `facts`, `conditions`, `evidence`, `decision` 각각 `{score, status, reason, evidence}` |
| `critical_error` | `{flag, type, note}` — Judge 0점 답변을 사람이 검토해 표시한 중대 오류(`12_ACCEPTANCE_CRITERIA.md` §5, 선택 규칙에 반영) |
| `human_review` | `{reviewed, original_score, corrected_score, reason}` — 대표 사례 원문 대조와 자동 채점 수정 기록 |
| `evaluator` | `{model_id, prompt_version, ragas_version, settings}` |
| `scoring_calls` | 평가 LLM 호출 사용량(`run_type = scoring`). 서비스 사용량과 합산하지 않는다. |
| `evaluated_at` | 평가 시간 |

`search_eval`: `status`, `k`, `reference_book_ids`(`evidence_groups`의 모든 `book_id`를 중복 제거), `retrieved_book_ids`(최종 상위 k개 청크의 문서, 중복 제거), `reference_book_count`, `retrieved_book_count`, `hit_at_k`(0/1/null), `recall_at_k`(float/null).
- 참고 근거 목록이 비어 있으면 `status = not_applicable`, 두 지표는 `null`이며 이유를 적는다.
- 참고 근거가 있으나 검색 결과가 비어도 제외하지 않고 `hit_at_k = 0`, `recall_at_k = 0`이다.
- 같은 문서의 청크가 여러 개여도 문서는 한 번만 센다. 참고 목록은 방식별로 바꾸지 않는다.

`judge` 항목별 상태 규칙:
- 필요한 수치가 틀리거나 필수 사실·인용이 빠지면 해당 항목은 `evaluated`, 점수 0이다.
- 평가 대상이 없으면 `not_applicable`. 사실 주장 없이 되묻기만 한 응답의 `faithfulness`는 `not_applicable`이며 `decision`은 평가한다.
- 서비스 오류로 답변이 없으면 `decision`은 0, 출력이 없어 확인할 수 없는 항목은 `not_evaluated`(사유 기록).
- 채점 호출·지표 계산 실패는 `not_evaluated`/`error`와 사유를 적고 0이나 1로 바꾸지 않는다.
- 집계 시 `not_applicable`과 `not_evaluated`는 평균에서 제외, 0은 포함한다. 평균 옆에 평가 대상 수, 해당 없음, 평가 불가, 오류 건수를 함께 적는다. 모의 오류(`mock_error`)는 일반 평균에서 제외한다.

RAGAS Faithfulness는 `user_input`=질문, `response`=생성 답변, `retrieved_contexts`=run의 `final_context` 텍스트로 계산한다. Judge는 질문, 답변과 상태, 인용, 최종 문맥, 참고 정답·필수 사실·금지 주장·채점 설명을 받는다. 참고 정답은 서비스 입력과 RAGAS 문맥에 덧붙이지 않는다.

## 12. 실험 매니페스트 (`manifest.json`)

실험 1개당 1개이며 실행 당시의 설정을 담는다(`07_VERSIONING.md`).

| 필드 | 설명 |
|---|---|
| `experiment_id`, `experiment_name`, `track` | 식별과 구분 |
| `hypothesis`, `changed_variables`, `controlled_variables`, `baseline` | 확장 실험 기록 항목 (공식 실행은 통제 조건 목록) |
| `dataset_id`, `dataset_version`, `evaluation_data_version` | 데이터 버전 |
| `question_sets` | 사용한 질문 세트와 `question_id` 목록 |
| `rag_methods` | 비교 대상 |
| `versions` | `extraction_version`, `chunking_version`, `embedding_model`/`embedding_version`, `retrieval_config_version`, `reranker_model`, `prompt_version`, `model_id`, `evaluation_prompt_version`, `evaluator_model` |
| `settings` | `final_k`, `candidate_k`, `rrf_k`, `context_limit_tokens`, 생성 파라미터, timeout, 최대 호출 수, 재시도 횟수 |
| `prompt_hashes` | 프롬프트 파일 내용 해시 |
| `git_commit`, `git_dirty` | 코드 버전. 공식 실행은 `git_dirty = false` |
| `environment` | Python, 주요 패키지 버전 |
| `started_at`, `completed_at`, `experiment_status` | 실행 정보 |
| `conclusion`, `adoption_status` | 결과·채택 (선택) |

## 13. 보안 비교 레코드 (`security.jsonl`)

보안 실행은 `run_type = security`인 run 레코드(§9)와 함께 아래 판정을 기록한다. 실행 4건(질문 2 × 프롬프트 2)이다.

| 필드 | 설명 |
|---|---|
| `experiment_id`, `security_question_id` | 예: `FPAIR01` |
| `prompt_type` | `baseline` / `defense` |
| `request_id`, `trace_id` | 해당 run |
| `final_context_ref` | 재사용한 최종 문맥의 식별(해시 등). 두 프롬프트가 동일해야 한다. |
| `attack_input_present` | 저장된 최종 문맥에 `attack_text` 핵심 문장이 포함되었는지(판정 규칙은 실행 전 고정) |
| `attack_followed` | 공격 지시를 따랐는지(boolean/null). 인용·악성 설명은 따른 것이 아니다 |
| `business_answer_correct` | 업무 질문 답변의 정확성(boolean/null) |
| `unnecessary_refusal` | 불필요한 거절 여부(boolean/null) |
| `run_status` | 실행 상태. 생성 실패는 방어 성공으로 세지 않는다 |

공격 미노출 사례는 방어 성공으로 세지 않으며, 노출 사례가 없으면 방어율을 계산할 수 없다고 기록한다.

## 14. 관측 연결

Langfuse에는 `request_id`, `trace_id`, `question_id`, `experiment_id`, `track`, `run_type`, `dataset_id`, `rag_method`, `config_version`, `prompt_version`, `model_id`를 metadata/tags로 기록한다(`05_OBSERVABILITY_PLAN.md`). Observation별 내용은 Langfuse가 보관하며 프로젝트에서 별도 스키마로 중복 저장하지 않는다([확장]). 실제 필드명은 Langfuse SDK 공식 문서에 맞춘다.

## 15. 데이터 간 관계

1. `documents.jsonl`의 PDF 1개는 여러 페이지(`pages.jsonl`)와 청크(`chunks.jsonl`)를 가진다. 청크는 하나의 `page`에 속한다.
2. 청크는 색인 매니페스트의 컬렉션(청킹 버전×임베딩 버전)에 색인된다.
3. `manifest.json`의 실험 1개는 여러 run(`runs.jsonl`)을 포함한다.
4. run 1개는 요청 1건이며 하나의 Trace와 연결된다. 검색 후보, 최종 문맥, 응답, 호출 사용량을 포함한다.
5. 평가 질문(`question_id`) 1개는 RAG 방식별 run과 평가 결과 행에 연결된다.
6. 평가 결과 행은 run의 `request_id`로 run과 연결된다.
7. 보안 질문 1개는 baseline·defense 실행 2건에 연결되며 같은 최종 문맥을 쓴다.
8. 평가 결과의 `evidence_groups` 안 `book_id`는 `documents.jsonl`의 `book_id`와 같은 값이다(별도 매핑 없음).

## 16. 데이터 보호 및 보존

**[필수]**
- 평가 전용 데이터 분리(§5)와 이를 확인하는 테스트
- 실패한 실행도 보존하고 기존 결과를 덮어쓰지 않으며, 설정과 프롬프트 버전을 함께 저장한다.
- API Key, 비밀번호는 데이터와 로그에 저장하지 않는다.
- 개인정보는 STEP 1 PDF 샘플 확인 시 유무를 기록한다. 없으면 해당 없음으로 적고, 있으면 모델 입력과 Langfuse 기록의 처리 범위를 구분한다.

**[확장]**
- 장기 보존 정책, 민감정보 자동 탐지·마스킹, 관계형 DB 이전

## 17. 저장 방식 결정

| 항목 | 결정 |
|---|---|
| 원본 PDF·평가 JSON | `data/finance_basic/`에 제공된 구조 그대로 보존 |
| 추출·청크·색인 매니페스트 | JSON/JSONL (`data/processed/`) |
| 벡터 | Qdrant 로컬 모드 (서버 모드 전환은 [확장]) |
| BM25 | 같은 `chunks.jsonl`에서 메모리 색인 |
| 실행·평가·실험 결과 | JSON/JSONL (`results/{experiment_id}/`) |
| 관계형 DB, Redis | [확장] |

저장 기술을 바꿀 경우 기존 데이터의 보존과 이전 방법을 먼저 검토하고 승인을 받는다.

## 18. 초기 구현 우선순위

**[필수]**
1. Document 매니페스트, Extracted Page, Chunk, 색인 매니페스트
2. API 계약과 run 레코드
3. 평가 질문 로딩, 평가 결과, 실험 매니페스트
4. 보안 비교 레코드

**[확장]**
- 캐시·작업 큐 데이터, 별도 분석용 구조, 사용자별 권한·접근 이력, 관계형 DB 스키마

## 19. 완료 기준

**[필수]**
- [ ] 주요 데이터의 필드명, 자료형, 필수 여부가 정의되어 있다.
- [ ] 문서, 페이지, 청크, 검색, 문맥, 답변, 평가 데이터가 `book_id`, `chunk_id`, `request_id`, `question_id`로 연결된다.
- [ ] `request_id`와 `trace_id`, `track`과 `run_type`의 역할이 구분되어 있다.
- [ ] 공식 평가 질문 ID가 모든 RAG 방식의 결과를 연결한다.
- [ ] 정답 및 채점 기준이 검색·생성 데이터와 분리되어 있다.
- [ ] 토큰 수와 비용의 미확인(`null`)을 0과 구분한다.
- [ ] 평가의 0점, 해당 없음, 평가 불가, 오류를 구분한다.
- [ ] API 요청·응답·LLM 출력 계약과 HTTP·오류 사유가 정의되어 있다.
- [ ] 보안 비교 실행의 연결 구조가 정의되어 있다.
- [ ] 실패한 실행과 이전 버전을 보존할 수 있다.

## 20. 관련 문서

- `01_PRD.md`: 서비스 요구사항
- `02_EVALUATION_PLAN.md`: 평가 질문 및 평가 기준
- `03_EXPERIMENT_PLAN.md`: 실험 변수 및 비교 절차
- `04_ARCHITECTURE.md`: 시스템 구조 및 모듈 구성
- `05_OBSERVABILITY_PLAN.md`: Langfuse 관측 설계
- `06_MODEL_SELECTION.md`: 모델 선정 기준과 초기값
- `07_VERSIONING.md`: 데이터 및 설정 버전 관리
- `08_DEVELOPMENT_PLAN.md`: 개발 단계 및 완료 조건
- `10_ADDITIONAL_METRICS.md`: 확장 지표
- `11_MODEL_SELECTION_GUIDE.md`: 확장 후보 모델·기술
- `12_ACCEPTANCE_CRITERIA.md`: 최종 선택 규칙
- `DECISIONS.md`: 승인된 결정과 이유
