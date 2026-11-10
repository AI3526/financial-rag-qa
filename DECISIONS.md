# DECISIONS.md

사용자가 승인한 결정, 선택 이유, 실제로 확인한 환경·버전을 기록한다. 형식은 `08_DEVELOPMENT_PLAN.md` §22.3을 따른다.

- 확인하지 않은 버전과 라이선스는 적지 않는다. 설치·다운로드 시점에 실제 값으로 채운다.
- 결정을 바꿀 때는 기존 항목을 지우지 않고 새 항목(상태: 변경)을 추가한다.
- 아래 D-001~D-019는 2026-11-06 설계 문서 검토에서 사용자가 승인한 결정이다.

---

## 1. 확인된 환경 (2026-11-06 확인)

| 항목 | 확인 값 | 비고 |
|---|---|---|
| OS | Windows 11 Pro 10.0.26200 | |
| CPU / RAM | Intel Core Ultra 9 275HX (24코어) / 약 31GB | |
| GPU | NVIDIA GeForce RTX 5060 Laptop | Blackwell. 최신 PyTorch(CUDA 12.8 이상) 필요 여부는 STEP 0에서 확인 |
| Python | 3.12.13 | `uv`가 관리하는 설치본. Windows Store `python` 별칭은 사용하지 않음 |
| uv | 0.11.32 | |
| Git | 설치됨 (버전 미확인) | 저장소는 아직 초기화되지 않음 → STEP 0에서 `git init` |
| Docker | 없음 | Qdrant 로컬 모드 사용(D-010) |
| Node | 없음 | 필요 없음 |

### 1.1 설치·확인된 패키지 (STEP 0, 2026-11-06, `uv.lock`으로 고정)

| 패키지 | 버전 | 비고 |
|---|---|---|
| openai | 3.46.0 | |
| langfuse | 5.2.0 | 환경 변수는 `LANGFUSE_BASE_URL` (코드 확인, `LANGFUSE_HOST`는 하위 호환) |
| langchain-core / langchain-openai | 1.7.6 / 1.7.4 | |
| pypdf | 6.21.4 | BSD-3-Clause (패키지 메타데이터 확인). PDF 추출(D-021) |
| python-dotenv | 1.2.4 | |
| pydantic (간접 의존) | 2.14.3 | |

Langfuse는 Cloud(`https://cloud.langfuse.com`)에 연결되며 인증 확인과 테스트 Trace 도착을 확인했다. `Langfuse.create_trace_id(seed=request_id)`로 `request_id`에서 trace_id를 결정적으로 만들 수 있음을 확인했다(같은 seed → 같은 값). 지정한 trace_id로 기록하려면 `trace_context={"trace_id": ...}`를 쓴다.

### 1.2 `gpt-6-luna` 확인 결과 (STEP 0, 2026-11-06, OpenAI SDK 3.46.0, Chat Completions)

| 항목 | 결과 |
|---|---|
| 모델 유형 | 추론(reasoning) 모델. 출력 토큰에 추론 토큰(`reasoning_tokens`)이 포함됨 |
| `temperature` | 기본값(1)만 허용. `temperature=0`은 400 오류 |
| `top_p` | 지원하지 않음 (400) |
| `seed` | 요청은 허용됨. 결정성이 실제로 보장되는지는 미확인 |
| `max_tokens` | 지원하지 않음. `max_completion_tokens`를 사용 |
| `reasoning_effort` | `none`, `low`, `medium`, `high`, `xhigh` 지원. `minimal`은 거부 |
| `response_format` | `json_object`, `json_schema`(strict) 모두 동작 |
| 스트리밍 | `stream_options={"include_usage": True}`로 사용량 집계 가능, 모델 기준 TTFT 측정 가능 |

영향과 후속 결정(STEP 8에서 실제 프롬프트로 확인해 동결):
- 같은 질문의 응답이 실행마다 달라질 수 있다(temperature를 낮출 수 없음). 결과의 한계로 기록한다. `seed`의 효과는 STEP 8에서 확인한다.
- `reasoning_effort`는 명시적으로 고정하고 기록한다(기본값에 의존하지 않음). 값은 STEP 8에서 제안한다.
- `max_completion_tokens`는 추론 토큰을 포함하므로 너무 작으면 답이 비는 경우(`finish_reason=length`)가 생긴다. 충분히 크게 잡고 잘림은 출력 오류로 처리한다. 값은 STEP 8에서 제안한다.
- LLM 출력은 `json_schema`(strict)로 형식을 강제할 수 있다. 다만 `source_ids`가 최종 문맥에 있는 ID인지는 코드가 따로 검증해야 한다.
- 토큰 사용량 기록에 `reasoning_tokens`를 별도 필드로 둔다(`09_DATA_SCHEMA.md`).

### 1.3 STEP 1에서 확인한 데이터 특성 (2026-11-06, Basic PDF 15개)

| 항목 | 확인 결과 |
|---|---|
| 규모 | 15개 × 24쪽 = 360쪽(과제 안내와 일치). 암호화·손상·이미지 없음, 모든 쪽에 텍스트(쪽당 평균 약 1,000자) → OCR 불필요 |
| 표 | 선(ruling) 기반 표 0개(pdfplumber 탐지). 숫자는 본문 문장 안에 있음. 표 처리 성능은 Basic에서 검증하지 못함 → Intermediate 확장 시 재비교 |
| 머리말 | FB01~08은 모든 쪽 첫 줄이 `FBxx \| 부서명`, FB09~15는 `FBxx`와 문서 제목만 있음 |
| 꼬리말 | 360쪽 모두 `N / 24`. PDF 페이지 번호와 일치함을 360쪽 전체에서 검증(테스트) |
| 줄바꿈 | 고정 폭 줄바꿈으로 단어 중간에서 끊김. 근거 원문과 대조한 줄 경계 134곳 중 87곳(65%)이 단어 중간, 47곳(35%)이 단어 사이 → 줄을 이을 때 공백 처리를 STEP 2에서 결정(빈 문자로 이으면 근거 원문과 정확 일치 24/64, 공백으로 이으면 9/64로 어느 쪽도 완벽하지 않음) |
| 개인정보 | 전화번호·주민번호·카드번호·URL 0건. 이메일 8건(FB11, FB15)은 모두 `.example` 도메인의 부서 공용 주소로 문서가 "실제 연락·인증에 사용할 수 없다"고 설명. 이름은 자동 검출하지 못했고 샘플로만 확인. **공개된 가상 연락처로 보고 개인정보 해당 없음(D-022)** |
| 대표 페이지 원문 대조 | FB10 p2(정정 공지: 아시아 280→260억, 유럽 240→260억)와 FB01 p6(수치·날짜 다수)를 PDF 렌더링 이미지와 비교해 수치·날짜·머리말·꼬리말이 모두 일치함을 확인(AI 보조 육안 대조) |

### 1.4 Langfuse 기능 조사 결과 (2026-11-06, 공식 문서 확인)

출처는 `https://langfuse.com/docs/...` 각 페이지이며, 문서에 없는 부분은 "확인 못함"으로 적는다.

| 기능 | 확인한 내용 | 우리 프로젝트에서의 의미 |
|---|---|---|
| Trace 이어붙이기 | `start_as_current_observation(trace_context={"trace_id", "parent_span_id"})`로 기존 Trace에 span을 붙일 수 있음. HTTP 전파(`traceparent`)와 Python의 자동 중첩은 해당 페이지에서 확인 못함 | 실험 Trace와 서비스 Trace의 연결 방식 설계에 영향(D-023 열린 점) |
| 외부 Trace의 데이터셋 항목 연결 | 문서 페이지에서 확인 못함. OpenTelemetry 속성으로 실험을 묶는 방법이 별도 페이지에 있음 | 사전 검증 필요(D-023) |
| 토큰·비용 | 전송한 `cost_details`가 우선, 없으면 모델 정의(Project Settings > Models, `match_pattern`+단가)로 계산. `usage_details`의 각 키는 **서로 겹치지 않는 구간**이어야 하며 `output`에서 `output_reasoning_tokens`를 빼야 이중 계산을 피함(OpenAI 사용량 스키마로 보내면 Langfuse가 자동으로 뺌) | `gpt-6-luna`는 `completion_tokens`에 추론 토큰이 포함됨. Langfuse로 보낼 때 이중 계산이 생기지 않게 처리 필요(STEP 8). 모델 단가가 확인되면 Models에 등록, 확인 전에는 비용을 계산하지 않음 |
| Scores / Score Analytics | 임의의 점수를 Trace·Observation에 붙일 수 있고, 사람 점수와 LLM Judge 점수의 일치도 분석 제공 | Hit/Recall/Faithfulness/Judge 점수를 같은 Trace에 붙이는 후보(STEP 11) |
| Annotation Queues | 도메인 전문가가 Trace·Observation·Session에 점수·코멘트·수정 출력을 남기는 검토 대기열. 데이터셋 실행 항목 추가는 문서에서 확인 못함 | "대표 자동 평가 결과를 사람이 원문과 대조", `human_review`·중대 오류 판정 작업에 맞음 [선택] |
| 관리형 LLM-as-a-Judge | Langfuse 워커가 평가 실행, 평가 모델은 LLM Connection, 평가 1회 비용 약 $0.01~0.10 추정(FAQ), 평가 실행도 `langfuse-llm-as-a-judge` 환경의 Trace가 됨. 내장 Faithfulness 템플릿은 없음. **Trace 단위 평가는 deprecated이며 2026-11-16 이후 Cloud에서 실행 중단**, Observation 단위 권장 | 과제 필수는 RAGAS Faithfulness 공식 예제 방식과 맞춤 Judge이므로 현행(자체 스크립트) 유지. 관리형을 쓴다면 Observation 단위로만 |
| Prompt Management | 프롬프트 버전 관리, SDK로 가져와 generation에 연결하면 버전별 지연·토큰·비용·점수 중앙값 비교. 폴백 프롬프트는 연결을 만들지 않음, Generation만 연결 | 프롬프트는 Git 파일로 버전 관리하기로 한 07번과 기준이 이중이 됨. STEP 8~9 프롬프트 조정 때 재검토 [보류] |
| Metrics API / 대시보드 | 비용·토큰·지연·점수 집계, `environment` 필터 가능. p50/p95와 태그·이름별 그룹은 해당 페이지에서 확인 못함 | 운영 지표의 기준 데이터는 결과 JSON(D-012). 대시보드는 시각화용 [선택] |
| Environments | 기록을 환경별로 구분, 화면·Metrics API 필터 | 도입함(D-022) |
| Masking | 클라이언트 초기화 때 `mask` 함수로 전송 전에 가림 | 개인정보 게이트(D-022) |
| (도입 결정) | 비용·Scores·Annotation Queues·Prompt Management·대시보드는 모두 도입하기로 함(D-024). 위 표의 [선택]·[보류] 표기는 도입 시점과 설계 결정이 남았다는 뜻으로 읽는다 | |
| 데이터 보존 | Hobby 플랜은 조회 가능한 기록이 30일, 이전 데이터는 저장되지만 업그레이드 시 조회 가능(요금 페이지) | **발표·제출에 필요한 Langfuse 화면은 30일 안에 캡처·링크 정리** 필요, Trace URL·내보내기로 보존 계획 필요 |

## 2. 확인 대기 항목 (실제 값 확인 후 채움)

| 항목 | 확인 시점 | 상태 |
|---|---|---|
| `reasoning_effort`, `max_completion_tokens`, `seed` 사용 여부의 최종값 | STEP 8 | 제안 대기 || PyTorch·sentence-transformers·qdrant-client·rank_bm25·kiwipiepy·ragas 등 나머지 패키지 버전 | 설치 시 (`uv.lock`으로 고정) | 미확인 |
| `intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`의 revision과 라이선스 | STEP 3·6 다운로드 시 모델 카드 확인 | 미확인 |
| 임베딩 소규모 확인 결과(Hit@5, Recall@5, 시간)와 승자 | STEP 3 | 미실행 |
| timeout, 최대 호출 수, 재시도 횟수의 값 | STEP 7~8에서 제안, 공식 실행 전 확정 | 미정 |
| PDF 추출 라이브러리, 청크 크기·overlap | STEP 1~2 | 미정 |
| 문맥 토큰 한도 최종값 | 청크 크기 확정 후 | 초기값 4,000 |
| 개인정보 유무 | STEP 1 PDF 샘플 확인 | 미확인 |

---

## 3. 결정 기록

각 항목: 결정 / 이전 상태 / 이유 / 영향 문서 / 승인.

### D-001 문서 검토 원칙
- **결정**: 과제 원본(`00_assignment`)을 기준으로 설계 문서 13개를 하나씩 검토하고, 승인된 변경만 반영한다. 원본은 수정하지 않고, `00_projectoutline.md` 하단 개인 메모도 요구사항으로 보지 않는다. `10_old/`는 참조하지 않는다.
- **이유**: 설계 문서가 아이데이션 중 일부만 수정되어 서로 불일치했음.
- **승인**: 사용자 (2026-11-06)

### D-002 상태 4종과 `error` 판정 주체
- **결정**: API 상태는 `answered` / `needs_clarification` / `abstained` / `error`. `error`는 LLM이 아니라 API 코드가 판정한다.
- **이전**: 여러 문서에 3개만 정의.
- **이유**: 과제 원본(step5)이 4종으로 구분하고, 자료 부족(`abstained`)과 시스템 문제를 구분해야 함.
- **영향**: 01, 04, 05, 09

### D-003 용어 통일
- **결정**: `book_id`(파일명에서 `.pdf`를 뺀 값, `document_id` 폐기), `question_id`(평가 JSON의 `id`), `track`(official/extension/development, 실험 단위), `run_type`(general_eval/security/mock_error/scoring, 요청 단위).
- **이유**: 과제 원본의 `book_id` 정의와 일치시키고, 식별자 역할 혼동을 막음(CLAUDE.md §4).
- **영향**: 02, 04, 05, 07, 08, 09

### D-004 필수/확장 범위 정리 (과설계 축소)
- **결정**: 필수는 `GET /health`, `POST /rag/query`, 정상·빈 질문·근거 부족·모의 모델 오류·timeout/재시도·Langfuse 장애 영향·평가 데이터 유입 방지 테스트. 멱등성 키, 동시성·부하, 서버 재시작, Liveness/Readiness 분리, `dataset` 파라미터, registry/interfaces 계층, Query Processing은 [확장].
- **이유**: 과제 필수 범위 대비 과설계였고, CLAUDE.md §3.1·§8이 불필요한 추상화를 금지함.
- **영향**: 04, 08

### D-005 일반 평가 질문 범위
- **결정**: development 30(튜닝용) / additional 30(공식 주결과) / expansion_validation 12 + 자체 질문(확인·확장용). 설정 동결 후 development+additional 60문항을 세 방식에 실행(답변 180개), 세트별 분리 보고, 비답변 8개 포함.
- **이유**: 어느 문서에도 질문 수가 정해져 있지 않았음(순환 위임). dev 60 + holdout 12는 표본이 작고 비답변이 없음. 정답이 공개 데이터라 완전한 holdout은 불가능하므로 튜닝에 쓴 문항을 분리 표시.
- **영향**: 02, 03, 08, 12

### D-006 질문 분류
- **결정**: Happy/Edge/Negative 분류를 도입하지 않고 `expected_action`(answer/clarify/abstain)만 사용한다. 질문 확장은 필수 완료 후 clarify·abstain 질문 추가로 한다.
- **이유**: 데이터에 해당 라벨이 없고 1:1 대응이 부정확함(예: 다문서 결합 answer 문항). 실제 분포는 각 세트 answer 26/clarify 2/abstain 2.
- **영향**: 02, 03, 09

### D-007 보안 공격 노출 판정
- **결정**: 저장된 최종 문맥에 `attack_text`(핵심 문장)가 포함되어 있으면 노출. 판정 규칙은 실행 전 고정. 보안 질문은 FPAIR01, FPAIR02.
- **이유**: 과제 원본은 노출 여부 기록을 요구하나 판정 방법이 없었음.
- **영향**: 02, 03, 09

### D-008 구현 순서
- **결정**: STEP 0→1→2→3→4→7→8→9(Naive RAG를 끝까지 연결)→5→6→10→11→12→13→14→15.
- **이전**: Dense, BM25/Hybrid, Rerank를 모두 만든 뒤 API·생성.
- **이유**: 과제 원본(step5)의 "질문 하나가 끝까지 동작한 뒤 세 방식을 같은 API에서 바꿔 실행" 안내. 통합 오류를 일찍 발견.
- **영향**: 04, 08

### D-009 모델·검색 초기 구성 (공식 실행 전 동결)
| 구성 | 초기값 | 이유 |
|---|---|---|
| 생성 모델 | `gpt-6-luna` (OpenAI API) | 과제 지정. 사용자가 API 키 보유 확인 |
| 임베딩 | `intfloat/multilingual-e5-large` vs `BAAI/bge-m3` 1회 비교 후 승자 동결 | e5는 과제 권장 계열(로컬, 다국어). BGE-M3는 입력 길이가 길어 표 청크에 유리할 수 있어 실측으로 비교. 선정 규칙(Recall@5 → Hit@5 → 시간)은 실행 전 고정 |
| 키워드 검색 | BM25(`rank_bm25`) + `kiwipiepy`, 같은 `chunks.jsonl`로 메모리 색인 | 청크가 1~2천 개 수준이라 충분하고 Qdrant 희소 벡터보다 단순. 토큰화 차이는 STEP 5에서 소수 질문으로 확인 |
| 결합 | RRF(k=60), 코드 구현 | 과제 권장. 점수 체계가 다른 두 검색을 정규화 없이 결합 |
| 리랭커 | `BAAI/bge-reranker-v2-m3` | 다국어(한국어 포함) 로컬 Cross-Encoder. Qwen3-Reranker는 [확장] |
| 후보 수 / 최종 k | Dense 20 + BM25 20 → RRF 상위 20 → (리랭크) → 최종 k=5 | 질문당 근거 문서가 1~3개라 k=5에서 Recall이 의미 있음 |
| 문맥 토큰 한도 | 4,000 (초기값) | k=5 청크에 지시문·질문을 더해도 충분. 청크 크기 확정 후 재확인 |
| 평가 모델 | `gpt-6-luna` (RAGAS, Judge 동일) | 과제가 허용, 예산·가용성. **한계: 자기 평가 편향 가능성** |
- **상태**: 초기값. 임베딩 승자와 정확한 버전·라이선스는 §2 확인 후 이 항목에 추가 기록한다.
- **영향**: 01, 04, 06, 08, 11

### D-010 Qdrant 실행 방식 (2026-11-06 수정: Cloud 우선)
- **결정**: Qdrant를 사용한다(과제 원본의 권장 구성). 실행 방식은 **Qdrant Cloud를 우선**하고 로컬(내장) 모드도 지원한다. `QDRANT_URL`과 `QDRANT_API_KEY`가 있으면 Cloud에 접속하고, 없으면 `QDRANT_PATH`의 로컬 폴더를 쓴다. 서버(Docker) 전환은 [확장].
- **이유**: 사용자가 Cloud 클러스터와 키를 준비했다. 로컬 모드는 한 프로세스만 폴더를 열 수 있어 STEP 7 이후 API 서버와 평가 스크립트를 동시에 쓰기 어렵다. 규모가 작아(청크 약 1천 개, 벡터 약 4MB) 어느 쪽이든 성능 문제는 없다. (이전 기록의 "원본은 Qdrant를 필수 도구로 요구하지 않음"은 오해를 부르는 표현이라 삭제. 원본은 e5·Qdrant를 권장하고, 필수 도구가 아니라는 것은 다른 선택도 허용한다는 뜻이다.)
- **확인한 것 (2026-11-06)**: 클러스터에 컬렉션이 0개(다른 프로젝트 데이터 없음). 무료 플랜의 용량·비활성 정지 정책은 **아직 확인하지 못함**. Cloud 첫 연결이 한 번 끊긴 적이 있어 호출에 재시도를 둔다.
- **영향**: 01, 04, 06, 07

### D-011 Langfuse 호스팅
- **결정**: Langfuse Cloud. 계정과 API 키(Public/Secret)는 사용자가 직접 발급해 `.env`에 입력한다.
- **이유**: Docker 없이 시작 가능하고 과제 데이터가 가상 문서·공개 평가셋이라 외부 전송 위험이 낮음.
- **영향**: 05, 08

### D-012 저장 구조
- **결정**: 파일 기반 JSON/JSONL. 요청 1건 = run 레코드 1개(검색 후보, 최종 문맥, 모델 입출력, `llm_calls[]`, 성능, 오류). 평가 결과는 별도 파일. ID 6종(`request_id`, `trace_id`, `experiment_id`, `question_id`, `chunk_id`, `book_id`). 관계형 DB, Observation 별도 저장은 [확장]. 결과 JSON이 비교표의 기준 데이터, Langfuse는 추적용.
- **이유**: 엔티티별 테이블·고유 ID(20종 가까이)는 필수 범위 대비 과설계였음.
- **영향**: 05, 09

### D-013 API 계약
- **결정**: `POST /rag/query` 요청 `{question, rag_method(dense|hybrid|hybrid_rerank), question_id?}`, 응답 `{request_id, status, answer, reason, citations[{source_id, book_id, file_name, page}]}`, LLM 출력 `{status, answer, reason, source_ids}`. 출처 ID(S1, S2…)는 코드가 부여·보관하고, 컨텍스트에 없는 ID는 오류 또는 제한 횟수 재생성. HTTP: 3상태 200 / 검증 실패 422 / 모델 오류 502 / 시간 초과 504 / 출력 검증 실패 502 / 검색·내부 오류 500. 오류율은 시스템 오류만.
- **이유**: 이전 문서들이 계약을 DATA_SCHEMA에 미루었으나 정의되어 있지 않았음. 사용자가 `/rag/query`·`rag_method` 표기를 선택.
- **영향**: 04, 08, 09

### D-014 청크 페이지 정책
- **결정**: 청크는 페이지 경계를 넘지 않고 단일 `page`를 가진다. 여러 페이지에 걸친 청크(`page_spans`)는 [확장].
- **이유**: 출처 표시가 단순하고 과제 원본 예시와 일치. 표가 페이지 경계에서 잘리는 문제는 STEP 1~2에서 실제 PDF 확인 후 재논의.
- **영향**: 08, 09

### D-015 버전 관리 단순화
- **결정**: 구성요소별 버전 체계 대신 실행 설정 스냅샷(`manifest.json`) 1개(버전 문자열, 모델 ID, 프롬프트 해시, `git_commit`, `git_dirty`, 호출 한도). PDF·추출 결과 sha256은 필수. 공식 실행은 Git 작업 트리가 clean일 때만 시작. Qdrant 컬렉션명에 버전 포함(예: `basic__chunk-v1__e5large-v1`), 기존 컬렉션 삭제 금지.
- **영향**: 07, 08, 09

### D-016 최종 선택 규칙 (12번 문서)
- **결정**: 합격선(게이트)이 아니라 사전 확정 선택 규칙. 항상 하나를 선택한다.
  - "품질이 비슷하다" = 주요 품질 지표 평균 차이가 모두 **±0.05 이내**. 이유: additional이 30문항이라 한 문제 차이가 0.033이며, ±0.03은 한 문제 차이에도 반응해 우연에 민감하고, 30문항의 평균 차이 신뢰구간은 약 ±0.1. (사용자 선택)
  - 지표 우선순위: ① Judge 사실·조건·판단 > ② Faithfulness·Judge 근거 > ③ Hit@k·Recall@k. 이유: 사용자가 받는 답의 정확성을 직접 재는 지표가 우선, Faithfulness는 정정 전 값을 충실히 따라도 높을 수 있고, 검색 지표는 중간 단계. (사용자 승인)
  - 중대 오류: 가장 적은 방식보다 **2건 이상 많으면 선택 후보에서 제외**, 1건 차이는 불리하게만 반영. 모든 방식에 있으면 가장 적은 방식을 선택하되 "실제 서비스 사용 불가"로 명시. (사용자 선택. 2건은 판정 오차를 감안한 제안값)
- **이전**: 합격선(Hit@5≥0.80 등)·동등성 검정·TTFT/TPS·Holdout 0건을 필수 게이트로 사용. 60문항으로는 검증 불가능하고 과제 원본에 합격선이 없어 변경.
- **영향**: 08, 09, 10, 12

### D-017 서비스 준비도 게이트
- **결정**: 합격선, TTFT/TTLT/TPS·p95, 오류율 1%·300회, 동등성 검정, Holdout 0건은 `10_ADDITIONAL_METRICS.md` §11의 [확장] 서비스 준비도 게이트로 이관. 수치는 초안이며 서비스 단계에서 근거를 검토해 확정.
- **영향**: 10, 12

### D-018 운영 지표 범위와 스트리밍 지표
- **결정**: 필수 운영 지표는 평균 응답시간·정상 건수, 오류율·실패 건수, 입출력 토큰 합계·확인 요청 수, (단가 확인 시) 비용. TTFT(`ttft_e2e`/`ttft_model`), TTLT(TTLS 폐기), TPS는 [확장].
- **영향**: 05, 10, 12

### D-019 프로젝트 관리 문서와 확장 방향
- **결정**: `DECISIONS.md`와 `TASKS.md`만 유지하고 `CHANGELOG.md`는 만들지 않는다(Git 이력으로 대체). Intermediate/Advanced와 실제 서비스(권한, UI, 문서 갱신, 캐시, 동시성)는 PRD §11 확장 로드맵에 정리하고 필수 범위에 영향을 주지 않는다. 필수 기록으로 PDF별 페이지 수·추출 성공/실패·추출 시간, 청크 수, 평균 청크 크기, 색인 시간을 채택.
- **영향**: 01, 08, 10

### D-020 저장소(Git/GitHub) 정책
- **결정**: 제출물이 GitHub 저장소이므로 저장소 하나로 관리한다. `project_docs/`, `CLAUDE.md`, `TASKS.md`, `DECISIONS.md`, `data/finance_basic/`(합성 데이터, 총 약 6.3MB)는 모두 커밋한다. `project_docs/00_assignment/`(과제 안내 복사본)는 공개 여부가 확인되지 않아 `.gitignore`로 제외한다(로컬에는 유지). `project_docs/10_old/`(이전 버전 설계 문서)도 현행 문서와 충돌하므로 제외한다(로컬에는 유지). `.env`, 가상환경, Qdrant 로컬 저장소, 모델 가중치·캐시는 제외한다. `data/processed/`(추출·청크 등 파생 데이터)도 원본과 같은 공개 범위로 커밋한다(원본이 이미 공개 저장소에 포함되므로 파생 데이터만 제외해도 보호가 되지 않음). `uv.lock`과 `pyproject.toml`은 커밋한다. `.env.example`은 변수 이름만 둔다(모델 이름 같은 비밀이 아닌 값은 채워도 됨).
- **이유**: 설계 문서와 커밋 이력은 "선택 규칙을 실험 전에 확정"했다는 증거이고 재현 가능성(과제 평가 관점)을 뒷받침한다. 문서를 `.gitignore`에 넣으면 로컬 이력도 사라진다(CLAUDE.md §8 위배). 데이터는 사용자가 합성 데이터라 올려도 무방하다고 확인했다.
- **주의**: 한 번 올린 파일은 이력에서 지우기 어려우므로 첫 커밋 전에 `.gitignore`를 적용한다. 첫 `git add` 전에 `.env`가 추적되지 않는지 확인한다.
- **승인**: 사용자 (2026-11-06)

### D-021 PDF 추출 라이브러리: pypdf
- **결정**: PDF 추출은 `pypdf`(6.21.4, 기본 `extract_text()`)를 사용한다. `pages.jsonl`에는 추출 텍스트를 가공하지 않고 저장하고(머리말·꼬리말·줄바꿈 포함), 줄 잇기와 머리말·꼬리말 처리는 청킹(STEP 2)에서 한다. 결과 파일은 덮어쓰지 않으며 재실행에는 `--force`가 필요하다.
- **비교 근거**: pypdf, pdfplumber 0.11.10, pypdfium2 5.14.0, PyMuPDF 2.1을 실제 360쪽에서 비교. 평가 데이터 `development.json`의 근거 원문 64개 포함률은 네 후보 모두 64/64(공백 무시 비교)로 **품질 차이 없음**. 공백을 유지한 비교도 네 후보 모두 9/64로 동일. 추출 시간은 pypdf 17초, pdfplumber 9초, pypdfium2 0.9초, PyMuPDF 0.8초. 이 비교는 추출 충실도 점검 용도였고 색인·생성에는 사용하지 않았으며 공식 평가용 `additional`은 사용하지 않았다.
- **이유**: 품질이 같으므로 라이선스·단순성으로 선택. pypdf는 과제 권장 도구, BSD-3-Clause, 순수 Python. 속도는 이 규모에서 문제 없음(Advanced 18,000쪽도 약 15분 추정, 실측 아님). PyMuPDF는 AGPL-3.0(또는 상용)이라 공개 저장소·재사용 시 조건이 붙을 수 있어 제외. pypdfium2는 `\r\n` 줄바꿈 정규화와 바이너리 의존이 필요하고, pdfplumber는 이번 데이터에서 이점(표)이 없고 의존성이 늘어남.
- **한계와 재검토**: Basic에는 표가 없어 표 처리 성능은 검증하지 못했다. Intermediate/Advanced 확장 시 pdfplumber·pypdfium2를 다시 비교한다([확장]).
- **승인**: 사용자 (2026-11-06)

### D-022 문서 추출 실행의 Langfuse 기록과 개인정보·마스킹 정책
- **결정 1 (Langfuse 기록)**: 추출 실행 1회를 Trace 1개(`document-ingestion`)로, PDF마다 Span 1개(`extract-pdf`)로 기록한다. Span 출력에 문서 요약(페이지 수, sha256, 상태, 시간, 오류)과 **페이지 텍스트까지 포함**한다. 질문 처리 Trace(`rag-query`)와 구분한다. Langfuse 실패는 추출을 막지 않는다(`--no-langfuse`로 오프라인 실행 가능). 이는 08번 §5의 [필수] 항목이었으나 STEP 1 첫 구현에서 누락되었다가 보완했다.
- **근거**: Langfuse Cloud 무료(Hobby) 플랜은 월 50,000유닛, 조회 30일, 사용자 2명(2026-11-06 공식 요금 페이지 확인). 유닛은 Trace·Observation·Score의 개수이며 텍스트 길이가 아니다. 페이지마다 Span을 만들지 않고 문서별 Span에 텍스트를 담아 한 번에 16유닛만 쓴다. 실제 실행으로 Trace 1 + Span 15, 페이지 360쪽, 텍스트 368,449자가 기록된 것을 API로 확인했다. 한도 초과 시 Hobby의 동작은 요금 페이지에 명시되어 있지 않아 확인하지 못했다.
- **(D-029 수정: 실험 Trace는 environment가 `sdk-experiment`로 고정되어 `track`은 태그·metadata로 구분)** **결정 1-1 (Span 이름과 환경)**: Span 이름에 원본 PDF 파일명을 넣는다(`extract-pdf FB01.pdf`)와 metadata에 `book_id`를 둔다. Langfuse 기본 필드 `environment`를 도입하고 `track`과 같은 값(`development` / `official` / `extension`)을 쓴다. 이에 따라 `development` 태그는 제거했다(태그는 `ingestion`, `connection_test` 같은 종류 구분에만 사용). 이름이 파일별로 달라져 대시보드의 이름 기준 집계는 쪼개지지만, 운영 지표의 기준 데이터는 결과 JSON이므로(D-012) 영향이 작다고 판단했다.
- **결정 2 (개인정보)**: 문서 속 이메일 8건(`.example` 도메인 부서 공용 주소, FB11·FB15)은 문서가 부서 공용 연락처이며 실제 연락·인증에 쓸 수 없다고 설명하는 **공개된 가상 연락처**로 보고 개인정보 해당 없음으로 한다. 마스킹하지 않으며 모델 입력·결과 JSON·Langfuse 기록 모두 원문이다.
- **이유**: 마스킹하면 "공시 문의 주소는?" 같은 질문에 `[EMAIL]`로 답하게 된다(평가 데이터에는 `@`가 없어 점수 영향은 없음).
- **저장소 공개 범위**: 이번 Basic 데이터는 합성 데이터라 원본(`data/finance_basic/`)과 파생 데이터(`data/processed/`)를 모두 공개 저장소에 포함한다. 원본이 이미 공개되므로 파생 데이터만 제외해도 보호가 되지 않는다.
- **정책 (게이트)**: **실제 개인정보가 포함된 데이터를 처리하기 전에**, ① `data/` 전체(원본과 파생 데이터, 결과)를 처음부터 `.gitignore`에 넣거나 비공개 저장소를 사용하고, ② Langfuse `mask` 함수(SDK가 클라이언트 초기화 때 지정하는 기능을 지원함을 설치된 소스에서 확인)를 `langfuse_client.py`에 구현·테스트하며, ③ OpenAI·Langfuse Cloud 같은 외부 서비스로 보내도 되는지 별도로 검토한다(법적 판단은 이 프로젝트에서 하지 않음). 이미 커밋한 파일은 이력에서 지우기 어려우므로 데이터를 넣기 전에 먼저 정한다. 그 전까지 실제 개인정보를 처리하지 않는다.
- **영향**: 05 §10.2, 08 §5, 09 §16
- **승인**: 사용자 (2026-11-06)

### D-023 Langfuse Datasets/Experiments 도입 시점과 규칙
- **결정**: Datasets/Experiments는 **STEP 11~12(평가·공식 비교)에서 도입**한다. 규칙은 지금 확정한다.
  1. 항목 `input`에는 **질문만** 둔다. 정답은 `expected_output`에, 필수 사실·금지 주장·근거 문서·`expected_action`은 metadata에 둔다. 서비스로는 `input`(질문)만 전달하며 테스트로 확인한다.
  2. 결과 JSON이 기준 데이터, Langfuse는 사본(보기·비교용)이다. 점수는 한 번 계산해 양쪽에 같은 값을 넣는다.
  3. 데이터셋은 평가 세트별로 나눈다(development, additional, expansion_validation). 보안 질문 2개는 방어 프롬프트 비교라 별도 구성으로 둔다.
  4. 공식 실행은 `run_name`에 실험 ID와 방식을 넣어 방식별 실행으로 구분한다.
  5. 서비스 코드 안에서 데이터셋을 읽지 않는다.
- **이유**: 세 방식을 질문별로 나란히 비교한다는 목적에 실험 비교 화면(기준 실행 지정, 평균 점수·비용·지연, 케이스별 나란히 보기, Trace 열기)이 맞다. 데이터셋 항목·실행은 유닛으로 세지 않지만(Billable Units 페이지) 실험 중 생기는 Trace·Observation·Score는 유닛이다(질문당 지표 7개 × 180건 ≈ 1,260유닛이 공식 실행 한 번에 추가).
- **열린 점 (D-029에서 해결: 실험 Trace에 서비스가 `trace_context`로 이어 붙음)**: 실험이 만드는 Trace와 우리 서비스가 만드는 Trace를 어떻게 이을지(서비스가 현재 Trace 안에 중첩, 또는 trace_id·parent_span_id를 전달해 이어붙임). 현재 설계(서비스가 `request_id`에서 만든 trace_id로 독립 Trace 생성)와 충돌할 수 있어, STEP 11 전에 작은 실험으로 검증하고 04 §8, 05 §4.2, 09 §4.1을 함께 조정한다.
- **승인**: 사용자 (2026-11-06)

### D-024 Langfuse 기능 도입 범위와 로컬(JSON)-클라우드 기록 규칙
- **결정 1 (도입)**: 비용 추적, Scores, Annotation Queues, Prompt Management, 대시보드를 모두 도입한다.

  | 기능 | 시점 | 선결 사항 |
  |---|---|---|
  | 비용 추적 | STEP 8(사용량 기록), 단가 확인 후 Models 등록 | `gpt-6-luna` 단가를 **아직 확인하지 못함**. 확인 전에는 비용을 계산하지 않고 확인 불가로 기록(과제 원칙). 이중 계산 방지를 위해 추론 토큰은 `output`에서 분리해 전송 |
  | Scores | STEP 11 | Hit/Recall/Faithfulness/Judge를 같은 Trace에 붙이는 방법을 spike로 확인 |
  | Annotation Queues | STEP 11 | 사람 점수·코멘트를 로컬로 내려받는 단계 필요(결정 2) |
  | Prompt Management | STEP 8~9 | **프롬프트 원본(master) 위치를 정해야 함**(아래 열린 결정) |
  | 대시보드 | STEP 14 | 보여 줄 지표 정의 |

- **결정 2 (기록 규칙)**:
  1. 서비스·평가가 만드는 기록(Trace, 모델 입출력, 사용량, Score)은 **같은 코드 경로에서 로컬 JSON과 Langfuse에 함께 기록**한다. Langfuse 기록 실패가 JSON 저장을 막지 않는다. 두 곳이 어긋나면 로컬 JSON이 기준이다(클라우드 장애, 무료 플랜 30일 조회 제한).
  2. **Langfuse 화면에서만 생기는 데이터**(Annotation 점수·코멘트, UI에서 수정한 프롬프트)는 **로컬로 내려받아 JSON/Git에 보관**하는 내보내기 단계를 둔다. 내려받지 않은 데이터는 클라우드에만 남은 것으로 본다.
  3. **편집 가능한 정의**(프롬프트, 데이터셋 항목)는 편집하는 곳을 하나로 정한다. 양쪽에서 수정하지 않는다.
  4. 발표·제출에 필요한 Langfuse 화면은 30일 안에 캡처하거나 Trace URL을 정리한다.
- **결정 3 (프롬프트 원본)**: **Git 파일이 프롬프트의 원본**이다. 스크립트로 Langfuse에 버전을 올려 Trace와 연결하고, 프롬프트는 Langfuse UI에서 수정하지 않는다(수정하면 Git과 어긋남). 공식 비교는 프롬프트를 동결하고 해시·`git_commit`을 기록해야 하며(07번), Langfuse 장애가 서비스에 영향을 주지 않게 한다. Langfuse 연결 제약(폴백 프롬프트는 연결을 만들지 않고 Generation만 연결됨)은 구현 시 고려한다. 07 §7과 04 §8은 STEP 8에서 함께 고친다.
- **Trace ID 설계(요청 Trace와 실험 Trace의 연결)**: 질문 요청 Trace(STEP 7 이후)에만 해당하는 문제라서, 추출·청킹·색인 같은 실행 단위 Trace(STEP 1~3)와 무관하다. **→ D-029에서 spike로 확인해 확정했다.**
- **영향**: 04 §8, 05 §4.2·§11, 07 §7, 09 §4.1·§9·§11
- **승인**: 사용자 (2026-11-06)

### D-025 청킹 방식 (chunk-v1)
- **결정**: `paragraph-pack`. 청크는 페이지 경계를 넘지 않는다.
  1. 줄 0(머리말 `FBxx | …`), 줄 1(문서 제목), 줄 2(소제목)는 `metadata`(`header`, `doc_title`, `section_title`)로 옮기고, 꼬리말 `N / M`은 버린다. 검색용 텍스트는 `retrieval_text()`가 `머리말 / 제목 / 소제목` 접두를 붙여 만들며 `chunk_text`는 본문만 둔다. 구조가 다른 페이지는 내용을 버리지 않고 `unexpected_page_structure` 경고를 남긴다.
  2. 문단 경계: 줄 길이가 페이지 최장 줄의 85% 미만이면 문단 끝으로 본다.
  3. 줄 이음: 빈 문자열로 잇고, 줄이 `.?!,;:)”’`로 끝나면 공백을 넣으며 다음 줄의 원래 앞 공백은 보존한다. 단어 중간에서 줄이 바뀌는 PDF라 공백을 넣으면 회사·제품명이 쪼개진다.
  4. 문단을 `max_chars=450`까지 묶는다. `overlap_units=0`, `token_count=None`(토크나이저는 STEP 3에서 확정).
  5. **`hard_max_chars=800` (승인된 설계에 없던 추가)**: 450자를 넘는 문단(1,929개 중 19개, 최대 659자)은 문장 단위로 쪼개지 않고 통째로 한 청크로 둔다. 쪼개면 development 근거 원문 64개 중 2개가 두 청크로 갈라졌다. 800자를 넘는 문단만 문장 단위로 나눈다(현재 데이터에는 없음).
- **기각한 대안**: kiwipiepy 띄어쓰기 보정(고유명사가 훼손되어 불채택, 경계에만 쓴 변형은 근거 원문 54/64 일치에 이름 15건 분리).
- **결과**: 청크 1,056개(평균 325.5자, 중앙 347, 최소 54, 최대 659), 경고 0건. 360쪽 전부 청킹, 페이지별 내용 손실·중복 없음, 근거 원문 64/64가 단일 청크 안에 포함(튜닝에는 쓰지 않은 점검용).
- **열린 점**: 크기·overlap 튜닝은 03 §6에서 **[확장]**이므로 필수 범위에서는 하지 않고 `chunk-v1`을 그대로 쓴다(STEP 4 시작 때 정정). 확장으로 할 경우 새 `chunking_version`과 새 실험 ID로 하고 development에서만 조정한다. 표가 페이지 경계에서 잘리는 문제는 Intermediate로 이월.
- **승인**: 사용자 (2026-11-06, 설계 제안 승인). `hard_max_chars=800`은 구현 중 추가된 항목으로 보고 후 사용자 승인 (2026-11-06).

### D-026 STEP 3 임베딩 선정 규칙의 세부 정의 (실행 전 고정)
- **고정 시점**: 두 후보 중 어느 쪽도 색인·질문 평가를 실행하기 전 (2026-11-06).
- **규칙** (06 §11.2의 순서 그대로): ① Recall@5가 큰 모델 → ② 같으면 Hit@5가 큰 모델 → ③ 같으면 시간이 짧은 모델.
- **세부 정의**
  1. 질문: `development` 30문항 전체. 지표 대상은 `evidence_groups`가 있는 28문항(`answer` 26 + `abstain` 2). 근거가 비어 있는 `clarify` 2문항은 N/A(02 §5.3).
  2. 지표 단위: **`book_id` 단위(02 §5.3, 공식)**. Recall@5는 28문항 평균, Hit@5는 28문항 중 Hit=1인 비율. `(book_id, page)` 단위 값은 참고로만 기록하고 승자 결정에 쓰지 않는다.
  3. 비교 기준값은 소수 9자리까지 같으면 동점으로 본다.
  4. 시간(③): `색인 임베딩 시간(청크 1,056개) + 질문 30개 임베딩 시간`의 합. 모델 로드·Qdrant 저장·네트워크 시간은 제외. 같은 PC·같은 GPU에서 측정.
  5. 검색: Dense Top-5, 코사인 거리, 벡터 L2 정규화. e5는 `passage: `/`query: ` 접두어 사용, BGE-M3는 접두어 없음. 입력 텍스트는 `retrieval_text`(머리말·제목·소제목 접두 + 본문)로 두 모델 동일.
  6. 질문은 검색 입력으로만 쓰고 정답·근거 원문·`expected_action`은 색인과 임베딩 입력에 넣지 않는다.
- **결과 보존**: 두 컬렉션과 비교 결과(`results/dev-embed-select-001/`)를 모두 남기고 삭제하지 않는다.
- **승인**: 사용자 (2026-11-06). 세부 정의에 대한 질의응답(아래 "근거 설명")을 거쳐 실행 전에 승인.
- **근거 설명 (사용자 질의 응답 기록)**
  - **clarify와 abstain**: clarify는 질문이 모호해 무엇을 찾을지 알 수 없을 때 사용자에게 추가 정보를 요청하는 행동(상태 `needs_clarification`)이고, abstain은 질문은 분명하지만 자료에 답이 없어 이유와 함께 답할 수 없다고 알리는 행동(상태 `abstained`)이다. 예: F-DEV-19 "해온의 이번 실적"(해온솔루션/해온소재, 기간, 항목이 모호) = clarify, F-DEV-20 "루미센서 E의 영업이익 정확한 금액"(제품별 영업이익 자료 없음) = abstain.
  - **clarify 2문항의 N/A 제외**: `evidence_groups`가 비어 있어(F-DEV-19, F-DEV-30) 02 §5.3 "참고 근거 목록이 비어 있으면 검색 지표는 N/A"에 따른다. 0점으로 넣으면 모델이 못 찾은 것처럼 점수가 깎여 부정확하다. abstain 2문항은 근거가 등록되어 있어 지표 대상에 포함한다(포함 여부에 대한 별도 규정은 없고 현 규칙대로 포함).
  - **시간에서 네트워크·모델 로드 제외**: 두 모델이 같은 Qdrant·같은 네트워크를 쓰므로 네트워크 시간은 모델 속도와 무관한 잡음이다(실제로 저장 시간이 e5 6.4초, BGE-M3 27.1초로 달랐으나 모델 차이가 아니라 그 시점 네트워크 상태 차이로 판단). 모델 로드에는 다운로드 시간이 섞여 있다(e5 309초, BGE-M3 405초). 단 이는 **모델 선정용** 시간이며, STEP 14 서비스 응답시간에는 Cloud 접속의 네트워크 시간이 포함되고 보고서에 그 점을 적어야 한다.
  - **임베딩 방식**: 토큰마다 문맥이 반영된 1024차원 벡터가 만들어지고 풀링으로 청크 벡터 하나로 줄인다. e5는 토큰 벡터 평균(mean pooling), BGE-M3는 맨 앞 `[CLS]` 토큰 벡터를 쓰는 것으로 모델 카드의 일반적 설명에 따라 이해했으나, 이번에 라이브러리 설정 파일로 직접 확인하지는 않았다. 이후 L2 정규화로 내적 = 코사인 유사도.
- **실행 결과 (development 30문항, Dense, Top-5, 지표 대상 28문항, `results/dev-embed-select-001/`)**

  | | e5-large | BGE-M3 |
  |---|---|---|
  | Recall@5 (공식, 문서 단위) | **0.7649** | 0.7619 |
  | Hit@5 (공식) | 0.8571 (24/28) | **0.9286** (26/28) |
  | 페이지 단위 Recall (참고) | 0.4607 | 0.4583 |
  | 페이지 단위 Hit (참고) | 0.5714 | 0.5714 |
  | 시간 (색인 임베딩 + 질문 임베딩) | 31.495초 (30.3 + 1.195) | 30.736초 (29.83 + 0.906) |

  - 규칙 적용 결과: **e5, 결정 기준 Recall@5**. Langfuse Trace `44c458373e32b046f5290221769a27cc`.
  - **해석상 주의**: Recall@5 차이(0.003)는 사실상 없다. Hit@5는 BGE-M3가 2문항 많다. 질문 8개에서 결과가 달랐고, F-DEV-23·25는 e5가 근거 문서를 하나도 못 찾았으나 BGE-M3는 찾았다. 반대로 여러 문서가 필요한 F-DEV-11·17·24 등은 e5가 더 많이 찾았다. 규칙의 순서(Recall이 Hit보다 먼저) 때문에 미세한 Recall 차이로 승자가 정해졌다. 28문항은 표본이 작아 차이가 우연일 수 있다.
  - 사용자는 "Hit@5 차이가 있어 BGE-M3도 괜찮지만 정한 규칙대로 한다"고 판단했다. 규칙을 결과를 본 뒤 바꾸지 않았다.
- **확장 실험 후보**: BGE-M3 컬렉션(`basic__chunk-v1__bgem3-v1`)과 결과를 보존한다. 이후 additional 세트 등 더 큰 표본, Hybrid·Rerank 조합에서의 재비교, 승자 기준에 Hit를 함께 쓰는 규칙 비교를 [확장] 실험으로 할 수 있다(공식 결과와 구분되는 새 실험 ID로).

### D-027 임베딩 모델 동결: multilingual-e5-large
- **결정**: 공식 설정의 임베딩 모델을 `intfloat/multilingual-e5-large`로 동결한다(D-026 규칙의 결과). 이후 변경은 새 실험 ID로만 한다.
- **확인한 값 (2026-11-06)**: revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`, 라이선스 mit(모델 카드 API 값), 차원 1024, 최대 입력 512토큰(청크 토큰 평균 193, 최대 374로 초과 0개), 접두어 `passage: `/`query: `, 코사인·L2 정규화. 컬렉션 `basic__chunk-v1__e5large-v1`(Qdrant Cloud, 1056/1056 저장, 탐침 검증 통과). 색인 임베딩 30.3초.
- **비교 후보 기록**: BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`, 라이선스 mit, 컬렉션 `basic__chunk-v1__bgem3-v1`(보존).
- **기록 정정**: e5 색인 직후 매니페스트의 revision이 null로 저장됐다(코드 결함, 수정함). HF 로컬 캐시의 스냅샷 해시로 값을 보정하고 파일에 보정 사실을 적었다. 색인에 쓴 가중치는 그 캐시 스냅샷이다.
- **사고 기록 (모델 캐시 위치)**: `.env`의 빈 줄 `HF_HOME=`가 빈 문자열로 읽혀 모델이 저장소 안 `./hub`(6.4GB)에 내려받아졌다(모델 로드 309초·405초의 원인). `src/config.py`의 `load_env()`가 빈 값의 `HF_HOME` 등을 제거하도록 고쳤고(테스트 포함), `hub/`와 라이브러리가 만드는 `.agent_harnesses.json`을 `.gitignore`에 추가했다. 색인에 쓴 가중치의 revision은 `./hub`의 `refs/main` 스냅샷으로 확인했다(e5는 기본 캐시와 같은 해시). `./hub` 폴더 삭제 여부는 사용자 결정 대기(삭제하지 않음). 앞으로는 기본 캐시(`C:\Users\kantl\.cache\huggingface\hub`)를 쓴다.
- **남은 것**: 청크의 `token_count`는 `chunks.jsonl`을 덮어쓰지 않기 위해 null로 두고, 토큰 통계(평균·최대)는 색인 매니페스트에 기록했다. 청크 크기 튜닝(STEP 4)은 development로만 한다. 모델이 정해졌으므로 크기를 토큰 기준으로 보는 것은 STEP 4에서 검토.
- **영향**: 06 §11.2, 07, 09 §7
- **승인**: 사용자 (2026-11-06)

### D-028 Dense 검색 반환 형식과 유사도 임계값 미적용
- **결정**: `DenseRetriever.retrieve(question, k)`가 `{rank(1부터), score, chunk_id, book_id, file_name, page, text, metadata}` 목록과 `durations_ms`(`embed`, `search`, `total`)를 돌려준다. Hybrid·Rerank·API가 이 형식을 이어받는다. 빈 질문과 k<1은 `InvalidQuery`(API 422 `invalid_request`), 저장소·임베딩 실패는 `RetrievalError`(API 500 `retrieval_error`)로 구분한다.
- **Langfuse 기록 (08 §5 반영)**: 처음에는 "요청 Trace 설계가 STEP 7 전 spike로 미뤄져 있으니 검색 단계는 기록하지 않는다"고 판단해 건너뛰려 했으나, 08 §5가 Dense Retrieval 단계의 "검색어, 검색 결과, 점수, 검색 시간" 기록을 함께 구현하도록 요구하므로 잘못된 판단이었다(STEP 1에서 지적받은 것과 같은 유형). 사용자 질문으로 바로잡았다. `DenseRetriever(langfuse=...)`가 검색마다 Span `dense-retrieve`(유형 `retriever`)에 질문·k·컬렉션·임베딩 모델·결과(순위, 점수, 청크, 출처)·단계별 시간을 기록한다. Span이라 요청 Trace 안에서 호출되면 자동으로 그 아래에 붙고, Trace ID 설계(STEP 7 전 spike)와 무관하다. Langfuse 실패는 검색을 막지 않는다.
- **동작 확인 기록**: `scripts/check_dense_retrieval.py`가 질문 유형별 8개를 실행해 `results/dense-check-001/`(`manifest.json`, `retrievals.jsonl`, track=development, 성능 지표 아님)에 저장하고 Langfuse Trace `2ecfabf82feab8bb296b83be767d7881`(루트 1 + 검색 Span 8)에 기록했다. Langfuse에서 읽어 Span 8개가 모두 루트 아래에 붙고 질문·Top-5·시간이 들어 있음을 확인했다.
- **유사도 임계값은 두지 않는다.** 설계에 없고, 실제 점수로도 구분이 어렵다. 같은 컬렉션에서 관련 질문의 1위 점수가 0.86 전후, 전혀 무관한 질문(저녁 메뉴)의 1위가 0.787이었다. e5 코사인 점수가 좁은 구간에 몰려 있어 임계값으로 "근거 없음"을 가르면 정상 질문이 잘릴 위험이 있다. 근거가 충분한지는 생성 단계(STEP 8~9)가 판단한다.
- **확인한 동작 (2026-11-06)**: "2026년 상반기 영업이익" 질문에서 1위가 FB01 p.2(영업이익 96억원)였고, 근거가 없는 질문(2027년 1분기 확정 매출)도 Top-5를 돌려준다. 첫 호출은 모델 로드 때문에 약 0.8초, 이후 약 0.16초(임베딩 약 15ms + Qdrant Cloud 검색 약 145ms). 이 시간에는 Cloud 네트워크가 포함된다(STEP 14 보고서에 명시).
- **발견한 결함**: `k or 기본값`이 k=0을 기본값으로 바꾸는 문제를 단위 테스트가 잡아 고쳤다.
- **승인**: 사용자 (2026-11-06, STEP 4 진행 승인). 청크 튜닝을 [확장]으로 두는 정정(D-025)도 사용자 승인. 동작 확인 Trace 8개는 삭제하지 않고 유지하기로 함(참조 보존, 증거 용도).

### D-029 요청 Trace와 실험 Trace의 연결, 환경 규칙, Langfuse 기록 범위 (spike 결과)
- **배경**: D-023·D-024의 열린 점(실험 Trace와 서비스 Trace 연결)을 STEP 7 전에 spike로 확인했다. 사용자는 비교 화면(Experiments)이 필요하다고 판단했다("다른 구성원에게 설명할 대시보드, Langfuse를 쓰는 김에 활용할 수 있는 것은 활용"). 과제 원본은 Langfuse 사용과 "요청 ID로 Langfuse 실행 기록과 연결"을 요구하며 Datasets/Experiments는 요구하지 않는다(우리가 추가로 도입한 기능).
- **spike 확인 (2026-11-09, SDK 5.2.0)**
  - SDK 코드(`client.py` 3017~3129행): `dataset.run_experiment(task=...)`가 항목마다 Trace(`experiment-item-run`)를 만들고, 실험·데이터셋 항목 연결은 별도 호출이 아니라 **Span 속성(propagated attributes)**으로 기록한다. 이미 있는 Trace를 데이터셋 항목에 연결하는 API는 없다.
  - 실행: 임시 데이터셋(항목 2개)을 `run_experiment` 2회(`spike-run-k5`, `spike-run-k2`) 실행. `task` 안에서 컨텍스트를 물려받지 않는 새 스레드가 `trace_id`와 부모 Span ID만 받아 `start_as_current_observation(trace_context=...)`로 서비스 Span을 만들었다. Langfuse에서 읽은 구조는 `experiment-item-run > experiment-item-task > service-request > dense-retrieve`로 서비스 Span이 같은 Trace 아래에 붙었고, 채점 함수의 `hit_at_k`, `recall_at_k`가 같은 Trace의 Score(출처 API)로 저장됐다. 사용자가 비교 화면을 직접 보고 원하던 화면이 맞다고 확인했다.
  - 흉내 낸 것: 실제 HTTP 호출이 아니라 새 스레드. HTTP 헤더 또는 요청 필드로 `trace_id`/부모 Span ID를 전달하는 부분은 STEP 7에서 확인한다.
  - 코드로 데이터셋 실행 목록을 읽는 호출은 찾지 못해(제가 쓴 메서드 이름이 틀렸음) 비교 화면 확인은 사용자의 눈으로 했다.
- **결정**
  1. **Trace 연결**: 평가 실행(공식·보안)은 실험이 Trace를 만들고, 서비스는 요청에 담긴 `trace_id`와 부모 Span ID로 그 아래에 이어 붙는다. 일반 요청은 서비스가 Trace를 만든다. `request_id`에서 `trace_id`를 미리 계산하던 설계(`create_trace_id(seed=request_id)`)는 폐기하고 **실제 기록된 `trace_id`를 run 기록에 저장**한다. 09 §4.1, 05 §4.2·§11을 고쳤다. `langfuse_client.py`의 `trace_id_for`, `new_request_ids`는 STEP 7에서 정리한다.
  2. **환경 규칙(D-022 수정)**: 실험이 만든 Span의 `environment`는 SDK가 `sdk-experiment`로 고정한다(spike에서 확인, 우리가 이어 붙인 Span은 `default`로 기록됨). 따라서 문서 처리·색인·일반 요청 Trace는 `environment = track`을 유지하고, **실험 Trace에서는 `track`을 태그와 metadata로 구분**한다. 05 environment 행에 반영했다.
  3. **점수 방식**: `run_experiment`의 `evaluators`는 지표마다 `EVALUATOR` Span이 생겨 유닛이 늘어난다. 기본은 실행 후 `create_score(trace_id=...)`로 점수를 붙이는 방식이고, 비교 화면에 똑같이 나오는지 STEP 10에서 확인한 뒤 확정한다. 확인 전까지 `evaluators` 방식을 대비책으로 둔다.
  4. **Langfuse 기록 범위(유닛 절약)**: 튜닝과 개발 실험은 **로컬 JSON에만 남기고**, **공식 실행(`official`)과 보안 비교만 Langfuse에 올린다.** 단계별 구현 확인(소량의 동작 확인, 08 §5의 계측 구현 검증, STEP 10 대표 요청 추적)은 08 §5에 따라 소량 기록한다. 로컬 JSON이 기준 데이터라는 D-024 규칙과 같은 방향이다.
- **유닛 추정(실측 아님)**: spike에서 항목 하나당 Trace 1개에 관측 7개(서비스 Span 포함)가 생겼다. 지표 7개를 `evaluators`로 내면 항목당 약 24유닛, 공식 실행 180건 한 번에 약 4,300유닛. 점수를 따로 붙이면 항목당 약 13~15유닛, 한 번에 약 2,300~2,700유닛으로 줄 것으로 추정한다. 무료 플랜 월 5만 유닛(D-024 §1.4 조사값). STEP 10에서 대표 요청 1건을 실제로 재서 180건 분량을 계산한 뒤 공식 실행을 시작한다.
- **임시 데이터셋**: `spike-trace-link`는 사용자 승인으로 삭제한다. SDK에 데이터셋 삭제 API가 없어(확인함) Langfuse 화면에서 사용자가 삭제한다. spike가 만든 실험 Trace 약 28개 관측은 남는다(30일 조회 제한으로 사라짐).
- **영향**: D-022(환경 규칙), D-023 열린 점(해결), D-024 Trace ID 설계(해결), 04 §8, 05 §4.2, 09 §4.1
- **승인**: 사용자 (2026-11-09)

### D-030 gpt-6-luna 지연 측정과 timeout·재시도 초기값 (초기값, 미확정)
- **측정 (2026-11-09, `scripts/measure_model_latency.py`, 결과 `results/model-latency-001/`)**: 실제 Dense Top-5 문맥(입력 약 1,100~1,270토큰)과 잠정 지시문(JSON 스키마 strict), `max_completion_tokens=4000`, `max_retries=0`으로 질문 3개 × `reasoning_effort` 4단계(none, low, medium, high) × 2회 = 24회 호출. 오류 0건. 로컬에만 저장(Langfuse 미기록, D-029).

  | reasoning_effort | 평균(초) | 최대(초) | 추론 토큰 |
  |---|---|---|---|
  | none | 2.26 | 3.85 | 0 |
  | low | 1.59 | 2.28 | 0 |
  | medium | 2.53 | 3.31 | 43~85 |
  | high | 3.24 | 3.86 | 0~149 |

  출력 토큰은 48~260. 같은 입력·같은 effort에서도 응답 시간이 최대 2배 차이 났다.
- **관찰**: 같은 입력을 같은 effort로 두 번 호출해도 `status`가 달랐다. 근거가 있는 질문(8회) 중 6회 `answered`, 2회 `needs_clarification`(low 1회, high 1회). temperature가 1로 고정이라 응답이 매번 달라질 수 있다(06 §의 한계 기록과 일치). 이 프롬프트는 잠정안이라 STEP 8~9에서 개선할 대상이며, 공식 비교의 한계로 기록할 점이다.
- **초기값 제안 (사용자 위임으로 적용, 확정 아님)**
  - 생성 모델 호출 timeout **30초**: 관찰 최대 3.9초의 약 8배. 최종 문맥(최대 4,000토큰)은 측정 입력(약 1,200토큰)의 3배 이상이고 꼬리 지연을 24회 표본으로 알 수 없어 여유를 두었다.
  - 재시도: 연결 오류·429·5xx만 최대 2회(간격 1초, 2초). 출력 검증 실패 재생성 1회. **요청당 모델 최대 호출 3회**(재시도와 재생성 합산).
  - Qdrant 검색 timeout 10초, 최대 3회(간격 1초, 2초). 실측 검색 약 0.15초.
  - Langfuse 기록은 재시도 없이 실패를 무시한다.
  - 세 RAG 방식에 동일하게 적용하고, 재시도한 시도의 시간·토큰도 같은 요청에 합산한다.
- **STEP 8 실제 프롬프트 재측정 (D-032, 6건)**: 입력 1,396~1,674토큰에서 모델 시간 1.76~2.32초(low). 초기값 30초는 그대로 두고, 꼬리 지연과 `max_completion_tokens` 영향은 STEP 9 이후 더 많은 호출로 다시 본다.
- **미확정인 이유**: 24회 표본은 꼬리 지연(드물게 느린 호출)을 보기에 작고, `reasoning_effort`·`max_completion_tokens`는 STEP 8에서 정한다. 공식 실행 전에 STEP 8의 실제 프롬프트로 다시 재서 확정하고 이 항목을 갱신한다.
- **영향**: 04 §13.4, 08 §11·§12
- **승인**: 사용자가 진행을 위임함(2026-11-09, "알아서 네 말대로 진행해"). 값 자체의 최종 확정은 공식 실행 전.

### D-031 STEP 7 FastAPI 구현 결정
- **구조**: `src/api/` — `pipeline.py`(검증→검색→생성→출력 검증→출처 변환, 오류 구분, 설정), `app.py`(`GET /health`, `POST /rag/query`, 오류 처리), `models.py`(Pydantic), `tracing.py`(요청 Trace), `recording.py`(run 레코드 저장), `main.py`(서버 조립, 팩토리).
- **생성기는 교체 가능한 인터페이스(`Generator`)**로 두고 STEP 8에서 gpt-6-luna를 연결한다. 연결 전 서버는 정상 질문에도 답을 만들어 내지 않고 `500 internal_error`를 반환하며, 이때 모델을 호출하지 않았으므로 `llm_calls`에 기록하지 않는다. 테스트는 가짜 생성기로 정상·모의 오류·timeout·재시도·출력 검증을 확인한다.
- **요청 본문 계약(09 §8.2)은 바꾸지 않았다.** 추가된 것은 선택 헤더뿐이다: `traceparent`(평가 실험 Trace에 이어 붙기, D-029), `X-Experiment-Id`(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, 저장 경로에 쓰이므로 엄격히 제한), `X-Track`(development/official/extension), `X-Run-Type`(general_eval/security/mock_error/scoring). 허용값 밖은 `422 invalid_request`. 응답 헤더 `X-Request-Id`, `X-Trace-Id`. 요청 본문의 평가용 필드(`reference_answer` 등)는 무시하며 검색·생성기에 전달하지 않는다(테스트로 확인).
- **오류 형식**: 입력 검증 실패(빈 질문, 지원하지 않는 `rag_method`, 필드 누락, 깨진 JSON, UTF-8이 아닌 본문)는 모두 `422 {request_id, status:"error", reason:"invalid_request"}`. 검색 500 `retrieval_error`, 모델 502/504, 출력 검증 502, 내부 500. 존재하지 않는 경로 등(404, 405)은 질문 처리 계약 밖이라 FastAPI 기본 응답을 유지한다. 예상하지 못한 예외는 내용 없이 `internal_error`.
- **재시도·상한(D-030 초기값)**: 모델 호출 최대 3회(재시도와 재생성 합산), 연결 오류·429·5xx·timeout은 최대 2회 재시도(간격 1초·2초), 출력 검증 실패는 1회 재생성. Qdrant는 timeout 10초·최대 3회(간격 1초·2초)로 줄였다(기존 60초·4회는 최악 4분 이상 걸릴 수 있었음). 모델 호출의 timeout 30초는 생성기가 적용한다(STEP 8).
- **run 레코드**: 09 §9 필드로 요청 1건당 1행, `results/{experiment_id}/runs.jsonl`(헤더가 없으면 `results/service/runs.jsonl`, Git 제외)에 **덧붙여 저장**한다. 실패·422 요청도 기록한다. 저장 실패와 Langfuse 실패는 응답에 영향이 없다.
- **정리**: D-029로 폐기된 `trace_id_for`, `new_request_ids`를 `langfuse_client.py`와 테스트에서 제거했다.
- **발견한 결함과 수정**: 실제 서버 호출에서 UTF-8이 아닌 본문이 FastAPI 기본 `400 {"detail":…}`로 반환되어 오류 공통 형식을 어겼다. 공통 형식(422)으로 처리하도록 고치고 테스트를 추가했다.
- **검증 (2026-11-09)**: 전체 테스트 113개 통과(API 단위 42, 실제 검색 통합 2 포함). 실제 uvicorn 서버를 띄워 HTTP로 정상 질문(생성기 미연결이라 500), 빈 질문, `hybrid`, 필드 누락, 깨진 JSON, UTF-8 아닌 본문, 잘못된 헤더, 경로 침투 시도(`../evil`), 없는 경로를 호출해 모두 의도한 상태 코드와 공통 형식을 확인했다(Langfuse는 끄고 실행).
- **HTTP 구간 확인 (2026-11-09, D-029의 남은 항목)**: 실제 uvicorn 서버(Langfuse 켬, 검색은 실제, 생성기만 가짜)를 띄우고, `lf.run_experiment`(데이터셋 없이 로컬 데이터 2건)의 `task`가 `traceparent: 00-{trace_id}-{span_id}-01`을 실어 `POST /rag/query`를 HTTP로 호출했다. 결과: ① 서버가 응답 헤더 `X-Trace-Id`로 돌려준 값이 두 건 모두 실험 Trace ID와 일치. ② Langfuse에서 읽은 구조는 `experiment-item-run > experiment-item-task > rag-query > dense-retrieve`로 서버 Span이 실험 Trace 아래에 붙음(`rag-query`의 부모 = `experiment-item-task`). ③ `rag-query`와 하위 `dense-retrieve`의 metadata에 `request_id`, `question_id`, `experiment_id`, `track`, `run_type`, `rag_method`가 기록됨(일반 요청은 값이 없는 `question_id`·`experiment_id`를 생략). ④ 헤더 없이 보낸 일반 요청은 새 Trace가 만들어지고 `environment=development`. ⑤ 서버 run 레코드의 `trace_id`가 Langfuse Trace와 일치하고 `langfuse_recorded=true`. 실험 Span의 `environment`는 SDK 고정값(`sdk-experiment`), 서버 Span은 `default`(D-029 기록 그대로).
- **확인하지 못한 것**: Langfuse 화면의 태그(`run_type` 태그)가 실제로 붙었는지(관측 조회에 태그 필드가 없어 읽지 못함), 실제 모델 호출 timeout 동작(STEP 8), 실험 비교 화면에서 서버 Span 포함 Trace가 정상 표시되는지(구조는 확인, 화면은 보지 않음).
- **승인**: 사용자가 계획대로 진행하도록 승인(2026-11-09).

### D-032 STEP 8 gpt-6-luna 생성·출처 구현 결정
- **호출 방식 (사용자 결정, 2026-11-09)**: LangChain(`ChatPromptTemplate` + `ChatOpenAI`)을 쓰고 먼저 짧게 검증한다. 검증 결과 모두 통과해 OpenAI SDK 직접 호출로 바꿀 필요가 없었다. 재시도는 파이프라인이 담당하므로 `max_retries=0`.
- **검증한 것 (실제 호출 약 9회, langchain-openai 1.7.4, langchain-core 1.7.6, openai 3.46.0)**
  - `reasoning_effort`, `max_completion_tokens`, `response_format=json_schema(strict)`, `seed`를 `gpt-6-luna`가 그대로 받는다. `usage_metadata`에서 입력·출력·**추론 토큰**(`output_token_details.reasoning`)과 `response_metadata`에서 `model_name`, `finish_reason`을 읽을 수 있다(medium에서 추론 토큰 32 확인).
  - 오류는 `openai` 예외의 하위 클래스로 와서 `isinstance`로 구분된다: timeout, 연결 오류, 400(잘못된 파라미터), 401(잘못된 키). 429·5xx는 강제로 만들 수 없어 클래스 계층(`RateLimitError`, `InternalServerError`)으로만 매핑했다(**실제 발생으로는 미검증**).
  - **예상 밖 동작**: 출력이 `max_completion_tokens`에서 잘리면 `finish_reason=length`가 아니라 **HTTP 400**(`code=None`, 메시지에 "max_tokens or model output limit was reached")으로 온다. 코드가 없어 **메시지 문구로만 구분**할 수 있으므로(문구가 바뀌면 깨질 수 있는 취약한 방식) 일반 400(재시도 불가 `model_error`)과 구분해 `output_validation_error`로 처리하고 재생성한다. `finish_reason=length`가 오는 경우도 같이 처리한다.
- **오류 매핑과 재시도 가능 여부**: timeout → 504 `model_timeout`(재시도), 연결 오류·429·5xx → 502 `model_error`(재시도), 400·401·403·404 등 → 502 `model_error`(**재시도 안 함**, 호출 1회), 잘림·JSON 아님·없는 출처 ID → 502 `output_validation_error`(재생성 1회). 오류 메시지에는 원본 예외 문구를 넣지 않고 종류와 상태 코드만 남긴다(키 일부가 섞일 수 있음). 이를 위해 `PipelineError`가 `retryable`을 인자로 받도록 바꿨다.
- **생성 설정 (제안, 세 방식 동일, 아직 동결 아님)**: `reasoning_effort=low`(D-030에서 평균 1.59초로 가장 빨랐음), `max_completion_tokens=1500`(측정한 출력 최대 260토큰의 약 6배), `seed=20261109`(임의 값, 재현성 보조일 뿐 응답이 매번 같지는 않음). **품질은 측정하지 못했다.** STEP 9에서 development 질문으로만 확인한 뒤 공식 실행 전에 동결한다.
- **최종 문맥(08 §12.2)**: 검색 결과 rank 순으로 출처 ID를 붙이고 토큰 한도 4,000(초기값)을 넘으면 **낮은 순위 청크를 통째로 제외**한다(청크를 자르지 않고, 더 작은 뒤 청크가 앞질러 들어가지 않음). 남은 청크는 S1부터 다시 번호를 매기고 제외 목록을 기록한다. 1위 청크 하나만으로 한도를 넘으면 자료 없음으로 오판하지 않도록 그대로 두고 `over_limit`을 표시한다. `final_context`(모델에 실제 전달)와 `retrieval`(1차 검색 결과)을 구분해 기록한다.
- **토큰 수는 추정이다**: gpt-6-luna의 공식 토크나이저를 확인하지 못해 `tiktoken o200k_base`(langchain-openai 의존성으로 이미 설치)로 센다. STEP 8 이전에 측정한 실제 `prompt_tokens`와 대조하니 추정이 실제보다 6.0~6.9% 작았다(3건: 1175→1099, 1101→1025, 1270→1194, 응답 형식 지시와 메시지 형식이 추정에 빠지기 때문으로 보이나 원인은 확인하지 못함). **e5 토크나이저로 세면 실제보다 14.9~19.5% 작아(946, 888, 1081) 더 어긋난다.** e5 토크나이저는 임베딩 모델용(STEP 3의 입력 한도 점검에 사용)이며, 한도를 적용할 대상은 생성 모델의 입력이라 쓰지 않는다. gpt-6-luna 토크나이저는 확인하지 못했으므로 추정치임을 기록에 명시하고, 정확한 값은 호출 후 API가 돌려주는 `prompt_tokens`로 `llm_calls`에 기록한다. 실제 최종 문맥은 추정 865~1,138토큰이라 k=5에서는 한도 4,000이 걸리지 않았다(한도는 안전장치로 동작).
- **프롬프트**: Git의 `prompts/answer_v1.json`이 원본이고 sha256과 `prompt_version`(`answer-v1`)을 run 레코드 `versions`에 기록한다. 잠정 초안이며 STEP 9에서 development 질문으로만 조정하고 바꾸면 새 버전 파일을 추가한다(기존 파일은 덮어쓰지 않음). 평가 정답·근거는 프롬프트에 들어가지 않는다(테스트로 확인). **Langfuse Prompt Management 연결은 STEP 9 이후로 미뤘다(사용자 결정).**
- **실행 확인 (2026-11-09, 실제 서버 + 실제 gpt-6-luna, Langfuse 끔)**: 대표 6개 질문이 모두 200과 유효한 출처로 응답했다: 영업이익(answered, FB01 p.2의 96억원), 루미센서 E(answered), 리브필름 기본형 재생 원료 비율(answered, 공개되지 않았다고 답함), 2027년 1분기 확정 매출(abstained), 모호한 "리브"(needs_clarification, 기본형/R 확인 요청), 두 회사 매출 비교(answered). 호출당 입력 1,396~1,674토큰, 출력 72~123토큰, 추론 0(low), 모델 시간 1.76~2.32초, 요청 전체 2.0~2.8초(첫 요청은 워밍업 포함). D-030의 timeout 30초는 이 실측의 약 13배 여유다.
- **원문 대조 (정정 기록)**: 처음에는 이를 "사람이 원문과 대조"했다고 적었으나 **틀린 표현이었다.** 실제로는 AI(Claude)가 답변과 인용 원문을 읽고 비교한 것이며 사람의 확인은 아직 없다. AI가 본 결과: 두 회사 매출 비교(FB14 p.21: 해온솔루션 1,280억원, 해온소재 860억원, 각 법인의 별도 기준)와 기본형 비율(FB05 p.1·p.14: 기본형은 비율을 공개하지 않았고 R의 30%나 0%를 쓰지 않는다)은 답변과 원문이 일치하는 것으로 읽혔다. 과제가 요구하는 "대표 사례를 사람이 원문과 대조"는 **사용자 확인이 필요한 미완료 항목**이며 STEP 11에서 대표 사례를 정해 사용자가 직접 확인한다. 숫자 자동 대조는 "답변의 숫자가 인용 원문에 있는가"만 보는 것이라 의미 정확성은 보장하지 않으며, 근거 부족 답변은 질문에 나온 연도를 되풀이해 "없음"으로 표시된다(정상).
- **Langfuse Generation 기록 확인 (2026-11-09, 사용자 승인으로 요청 2건)**: `langfuse.langchain.CallbackHandler`를 요청마다 새로 만들어 연결했다(`main.py`, Langfuse가 있을 때만). Langfuse에서 읽은 결과: 일반 요청은 `rag-query > [dense-retrieve, ChatOpenAI(GENERATION)]`, 실험 경유(HTTP, `traceparent`) 요청은 `experiment-item-run > experiment-item-task > rag-query > [dense-retrieve, ChatOpenAI(GENERATION)]`로 Generation이 `rag-query` 아래에 붙었다. Generation에는 모델명 `gpt-6-luna`, 실제 모델 입력(system·user 메시지)과 출력 원문, 사용량이 기록됐다. 실험 경유 요청의 `trace_id`가 응답 헤더와 일치했다.
- **발견 (캐시 토큰)**: Langfuse의 사용량이 `input 3 / cache_read 1,564 / output 75`로 나뉘어 있었다(로컬 기록 입력 1,567 = 3 + 1,564). 몇 분 전 같은 질문을 보내 OpenAI의 프롬프트 캐시가 적용된 것으로 **추정**한다(API 응답의 캐시 필드를 직접 보지는 않았고 Langfuse의 분할로 짐작함). 영향: ① Langfuse 화면의 `input`은 캐시되지 않은 일부만 보이므로 **운영 지표의 입력 토큰은 로컬 기록의 전체 입력(`input_tokens`)을 기준으로 한다.** ② 같은 질문·같은 문맥을 반복하면 캐시로 시간과 비용이 달라질 수 있어, 세 방식의 응답시간 비교에서 문맥이 같은 요청이 먼저 캐시되면 뒤 요청이 유리해질 수 있다(STEP 14 한계로 기록, 실행 순서와 캐시 영향을 분석할 것). ③ 이를 추적하도록 `llm_calls`에 `cached_input_tokens`를 추가했다(모르면 null). 가격 계산에서도 캐시 입력 단가가 다를 수 있다.
- **한계와 미검증**: ① 대표 6건뿐이라 정확도에 대한 증거가 아니다(평가는 STEP 11). ② 같은 입력에서 `status`가 달라질 수 있다(D-030 관찰). ③ gpt-6-luna 가격은 사용자가 가져왔고 D-033에 기록했다. ④ 429·5xx 실제 발생 시 동작은 미검증. ⑤ Langfuse 화면(UI)에서 Generation이 어떻게 보이는지는 읽기 확인만 했다.
- **영향**: 04 §4.6, 08 §12, 07 §7
- **승인**: 사용자 승인 계획(2026-11-09). 생성 설정 값은 STEP 9 확인 후 동결.

### D-033 gpt-6-luna 단가와 비용 계산
- **단가 (2026-11-09)**: 출처 https://developers.openai.com/api/docs/models/gpt-6-luna. 100만 토큰당(USD): 입력 $0.10, 캐시 입력 $0.01(일반 입력의 10%), 캐시 쓰기 $0.125(일반 입력의 1.25배), 출력 $0.50. 입력이 272K 토큰을 넘으면 전체 요청에 입력·캐시 2배, 출력 1.5배. 지역 처리 +10%, Batch·Flex는 Standard의 50%, Fast는 2배.
- **저장 위치**: Git의 `configs/pricing.json`(원본). `.env`에 두지 않는다 — 비밀이 아니고 비용 계산 결과를 재현하려면 어떤 단가로 계산했는지가 저장소에 남아야 하기 때문. 출처·확인 날짜·적용 조건(`service_tier=default`)도 함께 적었다. Langfuse에 모델 단가를 등록하는 일(D-024의 "비용 추적")은 이 파일을 원본으로 하고 아직 하지 않았다.
- **계산 (`src/generation/pricing.py`)**: `(일반 입력 × 입력 단가 + 캐시 읽기 × 캐시 단가 + 캐시 쓰기 × 쓰기 단가 + 출력 × 출력 단가) / 1M`, 일반 입력 = 입력 − 캐시 읽기 − 캐시 쓰기. 출력 토큰에 추론 토큰이 이미 포함되어 있어 따로 더하지 않는다. 요청의 비용은 모든 모델 호출(재시도 포함)의 합이다. run 레코드에 `cost`, `currency`, `price_source`, `cost_status`로 기록한다.
- **계산하지 않는 경우(null)**: 사용량을 모르는 호출(예: timeout), 응답의 `service_tier`가 단가 파일의 조건과 다른 호출, 입력이 272K 초과, 토큰 값이 모순된 호출. 일부만 모르면 `cost_status=partial`로 합계가 하한임을 표시한다. 모르는 값을 0으로 바꾸지 않는다.
- **확인한 것 (실제 응답, 2026-11-09)**: `service_tier='default'`이고(OpenAI가 Standard 처리에 쓰는 값으로 **이해하나 문서로 확인하지는 못함**), 응답에 `cached_tokens`, `cache_write_tokens` 필드가 있다. 그래서 `llm_calls`에 `cache_write_tokens`와 `service_tier`를 추가했다(`cached_input_tokens`는 D-032에서 추가). 실제 서버 요청 6건의 비용은 직접 계산한 값과 일치했다(예: 입력 1,567 중 1,564 캐시 읽기, 출력 75 → $0.00005344).
- **해석 주의**: 위 6건은 같은 프롬프트를 몇 분 안에 반복해 입력의 99.8%가 캐시에서 읽혔다. 그래서 **캐시가 없는 첫 호출의 비용(입력 1,567, 출력 75 기준 약 $0.00019)보다 훨씬 낮게 나온다.** 공식 비교의 비용·시간은 캐시 영향(실행 순서, 같은 문맥의 반복)을 함께 분석해야 한다(D-032 발견 참조, STEP 14). 비용 규모는 서비스 답변 기준으로 매우 작다(공식 실행 180건이 캐시 없이도 약 3~4센트로 **추정**). 평가 호출(Judge 등)은 별도다.
- **영향**: 09 §9, D-024(비용 추적), STEP 14 운영 지표
- **승인**: 사용자 (2026-11-09, 가격 파일 생성과 같은 커밋 포함 승인)

### D-034 STEP 9 질문 조건 확인·보류 점검과 프롬프트 조정 (answer-v2)
- **원칙**: 상태 판단은 별도 규칙 코드가 아니라 STEP 8의 프롬프트와 출력 스키마에서 LLM이 한다(08 §13). 이 단계는 development 질문으로 동작을 확인하고 프롬프트를 조정한다. 조정 횟수에는 제한을 두지 않는다(처음에 AI가 임의로 "최대 2회"를 정했으나 사용자가 제한을 없애기로 결정, 2026-11-09). 대신 다음을 지킨다: 조정은 development 질문과 예외 질문으로만 하고 additional은 보지 않는다, 정답 문장과 회사·제품 이름은 프롬프트에 넣지 않는다, **프롬프트 문구를 새로 만들거나 바꿀 때는 전문을 먼저 사용자에게 보여 주고 승인받는다**(이번에 이를 지키지 않아 지적받음), 버전마다 이유와 결과를 기록하고 공식 실행 전에 동결한다. 조정을 많이 할수록 development 30문항에 맞춰질 위험이 커지므로 development 결과는 공식 결과가 될 수 없다. 버전마다 새 파일을 만들고 결과를 따로 보존한다(`results/dev-prompt-answer-v{1,2}-001/`, `results/exception-check-answer-v{1,2}-001/`, 로컬 전용·Langfuse 미사용).
- **development 30문항 기본 점검 (answer-v1, `scripts/run_dev_questions.py`)**: 서비스에는 질문 문자열만 보내고 정답 파일은 응답 후 비교에만 읽었다. 기대 행동과 응답 상태가 **26/30 일치**: clarify 2/2, abstain 2/2, answer 22/26. 불일치 4건(F-DEV-17, 22, 24, 25)은 모두 `answer`인데 `abstained`.
- **불일치 원인은 프롬프트가 아니라 검색이다.** 4건 모두 등록된 근거 페이지가 최종 문맥에 하나도 없었고(0/2씩), 모델은 가져온 자료로 확인되는 부분만 답하고 나머지는 확인할 수 없다고 밝혔다(08 §9.3 부분 답변과 일치, 숫자를 지어내지 않음). 전체로 보면 `answer` 질문 중 근거 페이지가 문맥에 전부 있던 9개와 일부 있던 6개는 **15개 모두 `answered`**, 하나도 없던 11개 중 7개 `answered`·4개 `abstained`였다. 이 4건을 `answered`로 맞추도록 프롬프트를 바꾸면 내용이 없는 답을 만들게 하므로 하지 않았다. 해결 방향은 검색 개선(STEP 5·6의 Hybrid, Rerank)이며 세 방식 비교에서 확인할 점이다.
- **근거 페이지 없이 `answered`한 7건의 대략 점검**: 기준 정답의 숫자가 답변에 모두 있는 것이 5건(F-DEV-03, 09, 18, 21, 23), 숫자가 다른 2건은 읽어 보니 F-DEV-13은 확인되는 부분(해온소재 필름사업 620억원, 리브필름 R)만 답하고 해온솔루션 장비사업 매출은 확인되지 않는다고 구분했고, F-DEV-28은 정정 후 유럽 260억원을 답하고 정정 전 240억원을 현재 값으로 쓰지 않았다. 이는 AI가 읽은 점검이며 정식 평가(필수 사실·금지 주장)는 STEP 11이다. 비슷한 내용이 여러 문서에 반복되는 데이터 특성일 수 있으나 확인하지 않았다.
- **예외 질문 6종(08 §9.4, `scripts/run_exception_questions.py`)**: 질문과 기대 동작은 실행 전에 파일에 고정했다(직접 만든 질문, 정답 파일 미사용, 단 질문 유형은 development 정답의 패턴을 알고 만들었으므로 공식 결과로 쓰지 않는다). answer-v1: 5/6 허용. **EX-3 "작년 영업이익은 얼마였나요?"가 실패**: 회사도 기간도 모호한데 모델이 해온솔루션과 "2026년 상반기 전년 동기 100억원"을 임의로 골라 답했다(`answered`, 되묻지 않음). EX-2(두 회사 합산)는 합산 2,140억원을 제시하며 연결 실적이 아니라고 덧붙였다(기대 기준 경계).
- **answer-v2**: 규칙 4(모호한 질문의 되묻기 조건) 한 줄만 구체화했다. 대상이 질문에 밝혀져 있지 않은데 자료에 서로 다른 대상이 함께 나오는 경우, 기준 기간·항목이 빠진 경우, '작년·이번 분기·지금'처럼 기준 시점 표현이 자료의 기간과 정확히 대응하지 않는 경우를 예로 들고, 자료에 비슷한 값이 있다는 이유로 빠진 조건을 추측해 채우지 않도록 했다. 나머지 규칙과 사용자 메시지 템플릿은 v1과 같다(테스트로 한 줄만 다름을 확인). 정답이나 회사·제품 이름은 넣지 않았다.
- **answer-v2 결과**: 예외 질문 **6/6 허용** — EX-3은 `needs_clarification`(어느 회사·기간인지 되묻고 자료에 있는 것을 알려 줌), EX-2는 `abstained`(회사별 수치는 밝히되 합산 실적은 확인되지 않는다고 함). development 30문항은 **v1과 30/30 상태가 같고** 26/30 일치(같은 4건), 일반 질문에 불필요한 되묻기가 새로 생기지 않았다.
- **기본 프롬프트를 answer-v2로 바꿨다**(`src/generation/prompts.py`의 `DEFAULT_PROMPT`). 동결은 공식 실행 전에 한다. **사용자가 answer-v1과 answer-v2의 문구를 전문 확인하고 그대로 두기로 승인했다(2026-11-09).** 두 프롬프트는 승인 전에 AI가 먼저 작성·변경했고 사후에 승인받았다.
- **한계**: ① 질문당 1회씩만 실행했고 같은 입력에서 상태가 달라질 수 있다(D-030 관찰). 이번에 v1과 v2가 30/30 같았던 것도 우연일 수 있어 반복 실행으로 확인한 것은 아니다. ② 예외 질문은 6개뿐이고 응답 내용은 AI가 읽고 판단했으며 사람의 확인은 아직 없다. ③ 부분 답변(EX-6, F-DEV-13·17·22)의 `status` 값은 일관되지 않다(`answered` 또는 `abstained`). 08 §9.3은 상태 이름보다 답한 범위와 보류 이유가 중요하다고 하므로 규칙으로 고정하지 않았으나 평가(STEP 11)에서 상태 일치 지표에 영향을 줄 수 있다. ④ development는 튜닝에 쓴 질문이라 공식 결과가 아니다.
- **영향**: 08 §13, 02 §(expected_action), D-032(프롬프트), 03 §5(질문 유형별 비교)
- **승인**: 사용자 승인(2026-11-09, STEP 9 진행). 프롬프트 동결은 공식 실행 전.

### D-035 STEP 5 BM25 토큰화 후보와 선정 규칙 (실행 전 고정, 사용자 승인)
- **배경**: 06 §5.3은 BM25(`rank_bm25`) + 한국어 토큰화(`kiwipiepy`)를 초기값으로 하고 토큰화 세부 설정은 STEP 5에서 소수 질문으로 확인하라고 했다. 확인한 문제: 기본 `kiwipiepy`는 회사명 "해온솔루션"을 `하/어/오/ㄴ/솔루션`으로 쪼개 "해온"이 사라진다(STEP 2에서 띄어쓰기 보정이 고유명사를 망가뜨린 것과 같은 계열). 패키지: rank-bm25 0.2.2, kiwipiepy 0.24.1(모델 0.24.0), 로드 약 1초.
- **후보 4개** (`src/retrieval/tokenize.py`): ① `whitespace`(공백 단어, 기준선), ② `kiwi`(형태소 분석, 내용어만), ③ `kiwi_userdict`(② + `configs/bm25_userdict.txt`의 회사·제품명 6개를 고유명사로 등록), ④ `char2gram`(공백을 뺀 글자 2-gram). 사용자 사전의 6개 이름은 **문서에 실제로 나오는 이름**(해온솔루션 850회, 해온소재 752, 루미센서 302, 루미뷰 352, 리브필름 332, 리브리턴 216, 청크 파일 기준)이고 모두 문서 제목(doc_title)의 첫 단어로 확인된다(15개 문서 제목 중 6개 이름이 첫 단어이고 나머지 첫 단어는 지역분류, 공시, 보고 같은 일반 용어). **정정**: 처음에 "평가 데이터에서 가져오지 않았다"고 적었으나 부정확했다. 이름이 문서에 있다는 것은 확인했지만, 이름을 고른 AI는 평가 질문·답변 출력을 이미 본 상태였으므로 선택이 평가 데이터 노출과 무관하다고 말할 수 없다. 사람이 고른 사전이라는 점(새 문서에 맞지 않음)도 한계다. 자동 추출(`Kiwi.extract_words`)은 시험해 보니 숫자 조각(`0억원`, `60억원`)이 많이 나오고 `해온솔루션`, `해온소재`, `리브필름`이 빠져 대체할 수 없었다.
- **규칙 (실행 전 고정)**: development 30문항, BM25 단독 Top-5, 근거가 있는 28문항(`clarify` 2문항은 N/A). `book_id` 단위 **Recall@5가 큰 후보 → 같으면 Hit@5가 큰 후보 → 그래도 같으면 `kiwi_userdict`, `kiwi`, `char2gram`, `whitespace` 순**(설계 초기값인 형태소 분석을 우선). 같다는 판단은 소수 9자리 기준. `(book_id, page)` 단위는 참고로만 기록하고 승자 결정에 쓰지 않는다. BM25 파라미터는 `rank_bm25` 기본값(k1=1.5, b=0.75)을 바꾸지 않는다.
- **사용자 결정 (2026-11-09)**: 사용자 사전 후보(③)를 포함한 4개와 제외한 3개를 **같은 실행에서 모두 평가하고 두 경우의 승자를 따로 보고**한다(③ 제외 시 동점 우선순위는 `kiwi`, `char2gram`, `whitespace`). 최종 선택은 결과를 본 뒤 사용자가 한다. 규칙(지표 순서와 동점 처리)은 결과를 보기 전에 이 문단으로 고정한다.
- **입력**: 청크와 질문에 같은 토크나이저를 쓰고, BM25 색인 텍스트는 Dense와 같은 `retrieval_text`다. 질문 문자열만 검색에 쓰고 정답은 검색이 끝난 뒤 지표 계산에만 쓴다. 로컬에만 저장(Langfuse 미사용, D-029), 결과는 `results/dev-bm25-tokenizer-001/`.
- **Hybrid 설정은 조정하지 않는다**: 06 초기값(Dense 20 + BM25 20 → RRF k=60 → 상위 20 → 최종 k=5). 후보 수·RRF 상수 조정은 [확장](08 §5.4).
- **구현 중 발견한 결함**: `rank_bm25`는 질문 단어가 전체 청크의 절반 이상에 나오면 IDF가 0 이하가 되어 단어가 일치해도 점수가 0이다. 처음에는 점수가 0인 청크를 모두 버렸는데 단위 테스트가 이를 잡아, **일치 여부는 점수가 아니라 토큰 일치**로 판단하도록 고쳤다.
- **승인**: 사용자 (2026-11-09, 규칙과 "사전 포함·제외 두 결과를 모두 보고" 방식 승인).
- **실행 결과 (2026-11-09, development 28문항, BM25 단독 Top-5, `results/dev-bm25-tokenizer-001/`)**

  | 토크나이저 | Recall@5 (공식, 문서 단위) | Hit@5 | 페이지 단위 Hit / Recall (참고) | 색인 시간 |
  |---|---|---|---|---|
  | whitespace | **0.7530** | 0.8571 | 0.5000 / 0.3476 | 0.10초 |
  | kiwi | 0.6310 | 0.8214 | 0.5000 / 0.3476 | 3.79초 |
  | kiwi_userdict | 0.6548 | 0.8571 | **0.5714 / 0.3833** | 3.75초 |
  | char2gram | 0.6369 | 0.7500 | 0.5000 / 0.3714 | 0.06초 |

  - **규칙 적용 결과: 4개 모두와 사전 제외 3개 모두 승자는 `whitespace`(결정 기준 Recall@5).** 사전이 효과가 있었는지 보면 `kiwi_userdict`가 `kiwi`보다 Recall@5 0.024, 페이지 단위 Hit 0.071 높았지만 `whitespace`에는 미치지 못했다. 아직 동결하지 않았다(사용자 승인 대기).
  - **해석 주의**: ① `whitespace`와 `kiwi_userdict`는 28문항 중 10문항에서 문서 단위 Recall이 달랐고 `whitespace`가 7문항, `kiwi_userdict`가 3문항에서 나았다(차이는 질문마다 방향이 갈려 우연일 수 있는 크기). Hit@5는 두 후보가 같다(0.8571). ② **참고 지표인 페이지 단위는 `kiwi_userdict`가 가장 높다(Hit 0.5714 vs 0.5000).** 규칙상 승자 결정은 문서 단위만 쓰므로 `whitespace`이지만, 두 지표가 다른 후보를 가리킨다. ③ `whitespace`는 조사가 붙은 어절을 그대로 쓰는 가장 단순한 방식인데도 가장 높았다. 원인은 확인하지 못했다(문서가 합성이라 같은 표현이 반복되는 특성일 수 있으나 검증하지 않음). ④ BM25 단독 성능이며 Hybrid에서의 효과와 다를 수 있다. ⑤ 기준 참고: 같은 28문항에서 Dense(e5)는 Recall@5 0.7649, Hit@5 0.8571이라 BM25(whitespace)와 비슷한 수준이다.
  - 5개 미만을 돌려준 질문은 네 후보 모두 0개였다.
- **동결 (사용자 승인, 2026-11-09)**: BM25 토크나이저를 **`whitespace`로 동결**한다(규칙 결과 그대로, 사용자 의견은 "동결"). `SparseRetriever`의 기본값이다. 이후 변경은 새 실험 ID로 한다. `kiwipiepy`와 `configs/bm25_userdict.txt`는 현재 쓰이지 않으며(토크나이저 후보 비교 코드는 재현을 위해 보존), 의존성과 파일을 지울지는 사용자가 정한다.

### D-036 STEP 5 Hybrid 연결과 Dense·BM25·Hybrid 검색 비교 (development, 승자 선정 아님)
- **구현**: `rag_method=hybrid`를 `/rag/query`에 연결했다(`src/api/main.py`). 설정은 06 초기값 그대로: Dense 후보 20 + BM25 후보 20 → RRF(k=60) → 상위 20 → 최종 k=5, 검색 필터 없음, BM25 토크나이저 `whitespace`(D-035 동결). 최종 문맥의 k와 토큰 한도는 Dense와 같다(통합 테스트로 확인: 두 방식 모두 최종 청크 5개). run 레코드에 `retrieval.dense / sparse / fused`와 설정(RRF 상수, 후보 수, 토크나이저)을 기록한다. `hybrid_rerank`는 STEP 6 전이라 422다.
- **비교 (development 28문항, 최종 Top-5, `scripts/compare_retrieval.py`, `results/dev-retrieval-compare-001/`, 로컬 전용)**

  | 방식 | Hit@5 (문서) | Recall@5 (문서) | 페이지 단위 Hit / Recall (참고) | 문서 단위로 근거를 하나도 못 찾은 질문 |
  |---|---|---|---|---|
  | Dense(e5) | 0.8571 | **0.7649** | **0.5714 / 0.4607** | 4개 (F-DEV-03, 20, 23, 25) |
  | BM25(whitespace) | 0.8571 | 0.7530 | 0.5000 / 0.3476 | 4개 (F-DEV-02, 09, 14, 25) |
  | Hybrid | **0.9286** | 0.7530 | 0.5357 / 0.4071 | **2개** (F-DEV-03, 25) |

  - **Hybrid가 모든 면에서 낫지는 않다.** Hit@5와 "하나도 못 찾은 질문 수"는 가장 좋지만, Recall@5는 Dense(0.7649)보다 낮고 페이지 단위 지표도 Dense가 더 높다. 질문별로 Hybrid가 Dense보다 Recall이 높은 질문은 2개, 낮은 질문은 5개, 같은 질문은 21개였다(BM25 대비: 높음 4, 낮음 4, 같음 20). 즉 문서를 하나라도 찾는 능력(Hit)은 보완되지만 여러 문서를 모두 찾는 능력(Recall)은 개선되지 않았다.
  - **STEP 9의 보류 4건(Dense가 근거 페이지를 놓쳐 `abstained`한 질문)은 Hybrid로 해결되지 않았다.** 문서 단위 Recall은 F-DEV-17 dense 1.00 → hybrid 0.50, F-DEV-22 0.50 → 0.50(페이지는 0.00 → 0.50으로 처음 찾음), F-DEV-24 1.00 → 0.50, F-DEV-25 0.00 → 0.00. 이 4건은 근거가 두 문서에 걸친 다중 근거 질문이라 Top-5로는 모자랄 수 있으나 원인은 확인하지 않았다.
  - **한계**: 28문항이고 development는 토크나이저 선정에도 쓴 질문이라 공식 결과가 아니다. 위 차이는 질문 몇 개 단위여서 우연일 수 있다. 공식 비교는 STEP 12(additional 포함 60문항)에서 한다.
- **발견한 BM25 한계**: `whitespace` BM25에서는 질문의 요청 문구가 순위를 좌우할 수 있다. 예: "2026년 상반기 실적을 알려 주세요."에서 "상반기"는 청크 1,056개 중 513개에 나와 IDF가 0.057인 반면 "주세요"는 1개에만 있어 IDF가 6.6이고, 1위 청크는 "해 주세요…"라는 문장이었다(조사가 붙은 어절을 그대로 쓰고 불용어 처리가 없기 때문). Hybrid에서는 Dense가 이를 보완하지만, 불용어 처리나 조사 제거는 필수 범위 밖이며 [확장](검색 설정 조정)이다.
- **구현 중 발견한 결함**: `rank_bm25`는 질문 단어가 청크의 절반 이상에 나오면 IDF가 0 이하가 되어 단어가 일치해도 점수가 0이다. 점수가 0인 청크를 모두 버리던 처음 구현을 일치 여부 기준으로 고쳤다(D-035).
- **Langfuse 확인 (2026-11-09, 사용자 승인으로 요청 1건)**: 실제 서버에서 `hybrid` 요청을 보내 Langfuse에서 읽었다. 구조는 `rag-query > [hybrid-retrieve > [dense-retrieve, sparse-retrieve], ChatOpenAI(GENERATION)]`로 Dense·BM25 Span이 `hybrid-retrieve` 아래에, 검색 Span과 생성(Generation)이 `rag-query` 아래에 붙었다. `hybrid-retrieve`의 metadata에 방식·RRF(k=60)·후보 수(20/20/20)·토크나이저(whitespace)가 있고, output에 최종 청크 5개와 후보(dense 20, sparse 20, fused 20), 단계별 시간(dense 834ms, sparse 2ms, fusion 0.1ms)이 기록됐다. 이 요청의 Dense 시간이 길었던 것은 서버 기동 직후 첫 호출이라는 점과 Cloud 네트워크 때문으로 보이며 원인은 확인하지 않았다.
- **미검증**: 점수 정규화 없는 RRF의 가중치 영향, Hybrid의 운영 지표(응답시간·토큰) 비교(STEP 14).
- **영향**: 08 §9, 06 §5.4, 09 §9 `retrieval`
- **승인**: 사용자(2026-11-09, STEP 5 진행·토크나이저 동결). 비교는 기록용이며 설정 조정은 하지 않았다.

### D-037 STEP 6 Reranker와 Hybrid + Reranking (development 비교, 승자 선정 아님)
- **Reranker 확인 (06 §11.1 초기값 `BAAI/bge-reranker-v2-m3`, 후보 간 비교는 [확장]이라 하지 않음)**: Hugging Face 모델 카드 기준 라이선스 **apache-2.0**, revision **`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`**(로컬 캐시의 스냅샷과 일치), 가중치 2.27GB(safetensors), 다국어 Cross-Encoder(XLM-RoBERTa 계열). sentence-transformers `CrossEncoder`로 읽히고 RTX 5060(`cuda:0`)에서 동작하며 최대 입력 길이 8,192토큰, 활성화 함수는 Sigmoid(점수 0~1)다. 확인 호출: 관련 문장 0.993, 무관한 문장 0.00002. 처음 읽는 데 353초가 걸렸는데 대부분 다운로드 시간이다.
- **구현 (`src/retrieval/rerank.py`)**: Hybrid의 RRF 상위 20개를 Reranker로 재점수해 최종 5개를 쓴다(06 초기값, 설정 조정은 [확장]). Reranker 입력 텍스트는 임베딩·BM25와 같은 `retrieval_text`(머리말·제목·소제목 + 본문)다. 점수가 같으면 재정렬 전 순위가 앞선 쪽이 먼저다. 후보 밖의 청크는 최종 결과에 들어오지 않는다(테스트로 확인). `rag_method=hybrid_rerank`를 `/rag/query`에 연결했다. 세 방식(Dense, Hybrid, Hybrid+Rerank)이 같은 최종 청크 수 k=5와 같은 문맥 한도를 쓴다(통합 테스트로 확인).
- **오류 처리**: Reranker 실패는 검색 오류로 보고 **500 `retrieval_error`**이며 **Hybrid 순서로 몰래 대체하지 않는다**(방식 비교가 섞이지 않도록). 서버는 Reranker 모델을 시작할 때 한 번 읽고, 읽지 못하면 시작되지 않는다(조용히 빠지지 않음). 후보가 없으면 모델을 호출하지 않고 빈 결과를 돌려준다.
- **기록 (09 §9)**: `retrieval.reranked = [{chunk_id, rank_before, rank_after, score}]`(후보 20개 전체), `durations_ms.rerank`, `performance.reranking_duration_ms`, 설정에 `reranker`와 `reranker_revision`을 남긴다. 점수는 모델이 돌려준 값 그대로이며 다른 방식의 점수와 단순 비교하지 않는다.
- **비교 (development 28문항, 최종 Top-5, `scripts/compare_retrieval.py`, `results/dev-retrieval-compare-002/`, 로컬 전용; 001은 STEP 5의 3방식 결과로 보존)**

  | 방식 | Hit@5 (문서) | Recall@5 (문서) | 페이지 단위 Hit / Recall (참고) | 근거를 하나도 못 찾은 질문 | 30문항 검색 시간 |
  |---|---|---|---|---|---|
  | Dense(e5) | 0.8571 | 0.7649 | 0.5714 / 0.4607 | 4개 | 5.5초 |
  | BM25(whitespace) | 0.8571 | 0.7530 | 0.5000 / 0.3476 | 4개 | 0.06초 |
  | Hybrid | **0.9286** | 0.7530 | 0.5357 / 0.4071 | **2개** | 5.0초 |
  | Hybrid + Rerank | 0.8929 | **0.8095** | **0.6429 / 0.5071** | 3개 | 20.4초 |

  - **Hybrid + Rerank는 Recall@5와 페이지 단위 지표가 가장 높다**(Hybrid 대비 문서 단위 Recall이 높은 질문 6개, 낮은 질문 3개, 같은 질문 19개). 반면 Hit@5는 Hybrid(0.9286)보다 낮고 근거를 하나도 못 찾은 질문도 Hybrid(2개)보다 많다(3개: F-DEV-14, 20, 25). Dense와 비교하면 Recall이 높은 질문 4개, 낮은 질문 4개로 같은 수인데 평균은 더 높다(이득이 큰 질문이 있었음).
  - **STEP 9의 보류 4건**: F-DEV-17은 문서 단위 Recall이 Hybrid 0.50 → Rerank 1.00으로 개선(페이지도 0.50), F-DEV-22는 0.50 그대로, F-DEV-24는 0.50 그대로(Dense는 1.00이었음), F-DEV-25는 0.00 그대로. 즉 일부만 개선됐다.
  - **비용**: 재정렬이 질문당 약 0.5초를 더한다(30문항 검색 시간 5.0초 → 20.4초로 **추정**, 첫 호출의 GPU 워밍업 포함, 네트워크 영향 있음). 운영 지표의 정식 비교는 STEP 14에서 한다.
  - **한계**: development 28문항이라 공식 결과가 아니고 차이는 질문 몇 개 단위다. 값은 두 번 실행해 같았다(검색이 결정적). 공식 비교는 STEP 12(additional 포함 60문항)에서 한다. 3방식 어디에서도 근거를 못 찾은 질문(F-DEV-25)이 있다.
- **구현 중 발견한 결함**: 비교 스크립트의 결과 저장 단계에서 변수 정의가 빠져 오류로 멈췄고(숫자는 화면에 나왔으나 저장 전), 불완전하게 만들어진 `results/dev-retrieval-compare-002/`(manifest 하나뿐)를 지우고 고쳐서 다시 실행했다. 최종 결과는 위 표와 같다.
- **테스트 시간 문제와 수정**: Reranker 통합 테스트를 추가한 뒤 전체 테스트가 72초에서 609~701초로 느려졌고, 재정렬 테스트 3개가 각각 약 200초(단독 실행은 약 1초)였다. 통합 테스트 파일마다 e5 임베딩 모델(GPU 약 2.2GB)을 따로 읽어 8GB GPU 메모리가 찼기 때문으로 **추정**했고(GPU 메모리를 직접 측정하지는 않음), 모델을 `tests/integration/conftest.py`의 세션 fixture로 한 번만 읽어 공유하게 고치자 전체가 **58초**로 줄었다. 서버는 e5와 Reranker를 각각 한 번만 읽으므로(약 4.5GB) 이 문제의 영향이 없다.
- **Reranker 선정 방식 (정정 기록)**: Reranker는 실험으로 고른 것이 아니다. 06 §11.1의 초기값(`bge-reranker-v2-m3`)을 설계 검토에서 정했고(Qwen3-Reranker 등 후보 비교는 [확장]), STEP 6에서는 그 모델이 실제 환경에서 동작하는지(라이선스, revision, GPU, 점수 형태)만 확인했다. 비교한 것은 "재정렬을 하는 방식과 하지 않는 방식"(Hybrid 대 Hybrid+Rerank)이지 Reranker 모델끼리가 아니다.
- **Langfuse 확인 (2026-11-09, 사용자 승인으로 요청 1건)**: 실제 서버의 `hybrid_rerank` 요청을 Langfuse에서 읽었다. 구조는 `rag-query > [rerank-retrieve > hybrid-retrieve > [dense-retrieve, sparse-retrieve], ChatOpenAI(GENERATION)]`이다. `rerank-retrieve`의 metadata에 방식·Reranker 모델·revision·후보 수(20)가, output에 최종 청크 5개, 재정렬 전후 순위·점수 20개(예: 재정렬 전 10위 청크가 2위로), 단계별 시간(dense 860ms, 재정렬 474ms, 합계 1,337ms)이 기록됐다.
- **세 방식이 모두 약한 부분 (development 분석, 원인은 관찰에 근거한 해석이며 검증하지 않음)**: 세 방식 모두 문서 단위 Recall이 1.0이 아닌 질문은 28문항 중 5개(F-DEV-13, 14, 22, 25, 26), 셋 다 근거 문서를 하나도 못 찾은 질문은 F-DEV-25 하나다. ① **근거가 여러 문서·페이지에 걸친 질문에서 약하다.** 문서 단위 평균 Recall은 근거 문서 1개인 질문(15문항)에서 Dense 0.80, Hybrid 0.93, Hybrid+Rerank 0.93이지만 2개인 질문(11문항)에서는 0.73, 0.55, 0.68이다. 페이지 단위는 근거 1페이지(15문항) 0.60·0.53·0.60, 2페이지(11문항) 0.32·0.27·0.45로 더 낮다. ② **근거 문서는 찾아도 정확한 페이지를 놓치는 경우가 많다**(단일 근거 질문에서도 페이지 단위 Recall이 최고 0.60). ③ F-DEV-25("루미뷰와 리브리턴을 같은 사업처럼 설명해도 되나")는 근거가 두 제품 설명 문서(FB03, FB06)의 1쪽인데 세 방식 모두 비교·사업부문을 설명하는 안내 문서(FB13, FB15, FB09)만 가져왔다(질문의 단어와 겹치는 안내 문서가 제품 문서보다 먼저 선택된 것으로 보이나 확인하지 않음). Hybrid는 다중 근거 질문에서 Dense보다 낮은 값이 많다(2개 문서 0.55 대 0.73).
- **미검증**: Reranker의 CPU 실행 시 지연, 입력이 길 때(문맥 한도) 동작.
- **영향**: 08 §10, 06 §6·§11.1, 09 §9
- **승인**: 사용자(2026-11-09, STEP 6 진행). 설정 조정은 하지 않았다.

### D-038 STEP 10 Langfuse 통합 검증 결과와 계측 보강
- **보강한 계측 (05 §5 필수 항목 중 빠져 있던 것)**: ① `context-construction` Span으로 **검색 결과와 최종 문맥을 구분**(검색된 청크, 최종 청크와 출처 ID 대응, 제외된 청크, 추정 토큰 수). ② `citation-and-status` Span(출력 검증 통과 후 출처 변환·상태). ③ 생성 호출 이름을 `answer-generation`으로(LangChain `run_name`), 재시도마다 Generation이 하나씩 생긴다. ④ `rag-query` output에 실행시간(`performance`)·비용·**오류 유형과 발생 단계**(`error.error_stage`, 상태 메시지 `reason (stage=…)`), ⑤ 필수 식별 정보를 Trace metadata에 추가: `dataset_id`, `dataset_version`, `chunking_version`, `embedding_version`, `prompt_version`, `prompt_sha256`, `config_version`. 태그는 `run_type:`, `track:`, `rag_method:` 접두어 형태. ⑥ 요청 형식 오류(필드 누락·깨진 JSON)도 Trace에 오류로 남긴다.
- **`config_version` 정의 (신규 정의, 사용자 승인)**: `cfg-` + 데이터셋·청킹·임베딩 버전, 프롬프트 sha256, 모델명, 파이프라인 설정(`PipelineSettings` 전체), 선택한 검색 방식의 설정을 정렬한 JSON의 sha256 앞 10자리. 같은 설정이면 같은 값이고 하나라도 바뀌면 달라진다(테스트로 확인). 05 §4.1에 정의를 적었다. 07의 실행 설정 스냅샷(manifest)과 별개이며 공식 실행에서는 manifest에 함께 기록한다.
- **Observation 이름 확정**: 05 §3.4의 예시 이름 대신 구현 이름으로 표를 고쳤다(`rag-query`, `dense-retrieve`, `sparse-retrieve`, `hybrid-retrieve`, `rerank-retrieve`, `context-construction`, `answer-generation`, `citation-and-status`, 평가 호출은 `faithfulness-evaluation`·`judge-evaluation`).
- **점검 도구**: `src/observability/trace_checks.py`(단위 테스트 15개: 정상 Trace는 모두 통과하고 일부러 망가뜨린 Trace는 정확히 해당 항목에서 실패)와 `scripts/verify_langfuse.py`. 실제 구성(Qdrant·e5·Reranker·gpt-6-luna)으로 요청 7건을 보내 로컬 run 레코드와 Langfuse의 Trace를 항목별로 대조한다.
- **점검 결과 (2026-11-09, 사용자 승인으로 요청 7건, `results/langfuse-verify-001/`)**: 점검 대상 6건(Dense 되묻기, Hybrid 답변, Hybrid+Rerank 답변, 모호한 질문 되묻기, 모델 호출 시간 초과 504, 빈 질문 422)이 **모든 검사(요청당 15~24개)를 통과**했다. 확인된 것: 요청 하나에 루트 Trace 하나이고 모든 관측이 그 아래에 연결됨, Request ID·Trace ID·`config_version` 일치(로컬 기록 대 Langfuse), 방식별 검색 Span 구조, 최종 문맥 청크가 로컬 기록과 일치하고 본문이 모델 입력에 들어 있음, 모델 입력·출력·모델명 기록, 사용량(입력 토큰 = 캐시되지 않은 입력 + 캐시 읽기, 출력 토큰)이 로컬 기록과 일치, 출처 변환 결과가 로컬 응답과 일치, 오류의 유형·발생 단계·재시도 횟수 기록(시간 초과 요청은 Generation 3개가 모두 ERROR로 표시되어 로컬 `llm_calls` 3건과 일치), 요청 6건의 Trace ID가 서로 다르고 누락 없음.
- **발견 1 — 비밀 값 검사 (정정 포함)**: 처음 검사가 모든 요청에서 "비밀 값 발견"으로 실패했다. 값은 출력하지 않고 위치만 확인한 결과 발견된 것은 **`LANGFUSE_PUBLIC_KEY` 하나뿐**이고 위치는 모든 관측의 `metadata.scope.attributes.public_key`로, **Langfuse SDK가 스스로 넣는 값**이다. OpenAI 키, Langfuse 비밀 키, Qdrant 키는 어디에서도 발견되지 않았다. 공개 키는 이름 그대로 공개 식별자라 비밀 값 실패에서 빼고 SDK가 넣는 위치에 있는지만 확인·별도 보고하도록 검사를 고쳤다. 이를 숨기려는 SDK 설정은 확인하지 않았다. **다만 "API 키가 기록되지 않는다"는 요구와의 관계는 사용자가 판단할 일**이다(공개 키만으로는 기록을 쓸 수 없다고 이해하나 직접 확인하지는 않음). 이 검사에서 같은 키가 두 패턴에 걸려 두 번 세어지던 결함도 함께 고쳤다(처음 보고된 "10건" 등은 중복으로 부풀려진 값). 처음 실패한 결과는 `report.json`에, 고친 검사로 다시 점검한 결과는 `report_recheck.json`에 보존했다.
- **발견 2 — 요청 형식 오류의 실행 구분 누락**: 필드 누락 같은 요청 형식 오류(7번째 요청)는 `X-Run-Type`·`X-Experiment-Id` 헤더가 적용되지 않아 `run_type=general_eval`·실험 ID 없음으로 `results/service/`에 기록됐다. 모의 오류가 일반 평가 오류율에 섞일 수 있는 결함이라 헤더 해석을 한 함수(`run_context`)로 합쳐 두 경로가 같은 규칙을 쓰게 고쳤다(단위 테스트 3개 추가). **이 7번째 요청은 수정 전에 보냈기 때문에 이번 점검 대상이 아니다.** Langfuse의 `development` 환경에 `run_type:general_eval`로 기록된 오류 Trace 1개가 남아 있다(공식 실행은 `official` 환경이라 영향 없음). 수정 후 경로는 단위 테스트로만 확인했고 실제 Langfuse로는 다시 보내 확인하지 않았다.
- **아직 확인하지 못한 것**: ① **평가 LLM 호출과 서비스 답변 생성 기록의 구분**(08 §14, 05 §15): 평가 호출이 STEP 11에서 만들어지므로 그때 확인한다 → **Faithfulness 평가 호출은 STEP 11에서 확인함(D-039)**, Judge 호출은 Judge 구현 후 같은 방식으로 확인한다. 규칙은 지금 정했다 — 평가 호출은 `faithfulness-evaluation`·`judge-evaluation` Generation, 태그 `run_type:scoring`, `rag-query` 아래에 두지 않고 서비스 `answer-generation`과 이름·태그로 구분한다. ② Langfuse 화면(UI)에서 대표 요청이 한눈에 보이는지(읽기 확인만 했고 화면은 사용자가 봐야 한다). ③ 보안 비교 4개 답변의 실행 기록(STEP 13), 실험별 설정·버전 연결(STEP 12). ④ 검사는 로컬 기록과 Langfuse가 일치하는지를 보는 것이어서, 둘 다 같은 오류를 갖는 경우는 잡지 못한다.
- **유닛 실측 (D-029의 남은 항목, 일부)**: 요청 하나가 만드는 관측 수를 Langfuse에서 세었다: Dense 5개(`rag-query`, `dense-retrieve`, `context-construction`, `answer-generation`, `citation-and-status`), Hybrid 7개, Hybrid+Rerank 8개, 모델 시간 초과(재시도 3회) 6개, 빈 질문 1개. 이를 공식 실행(60문항 × 3방식 = 180건)에 대입하면 관측 약 1,200개 + Trace 180개이고, 실험이 항목마다 만드는 관측 2개(`experiment-item-run`, `experiment-item-task`) 약 360개와 점수 약 1,260개(질문당 지표 7개 가정)를 더하면 **한 번에 약 3,000유닛(추정)**이다(D-029의 추정 2,300~2,700보다 약간 많음). Langfuse가 유닛을 어떻게 세는지(Trace 포함 여부 등)는 Billable Units 페이지로 확인하지 않았다. 점수를 `create_score`로 나중에 붙이는 방식과 비교 화면 표시 확인은 점수가 생기는 STEP 11에서 한다.
- **대표 요청 추적**: Hybrid+Rerank 요청(`d3ddb186…`)의 Trace `3b014688ea8b4090c47b7071b115ab38`에서 질문 → 검색(Dense·BM25 → 결합 → 재정렬) → 최종 문맥 → 모델 입출력과 사용량 → 출처·상태 → 최종 응답이 하나의 Trace에 있다.
- **영향**: 05 §3.4·§4.1·§15, 08 §14
- **공개 키의 출처 (SDK 소스 확인)**: Langfuse SDK가 추적 도구를 만들 때 공개 키를 계측 범위 속성으로 붙이고(`resource_manager.py`의 `attributes={"public_key": ...}`), 내보내기 필터가 이 값으로 "자기 프로젝트의 Span"만 골라 전송한다(`span_processor.py`). 이 속성이 OpenTelemetry 형식으로 전달되어 서버가 관측 metadata의 `scope.attributes.public_key`로 저장한다. 우리 코드가 넣은 값이 아니라 SDK 기본 동작이다. 이 속성을 끄는 설정이 있는지는 찾아보지 않았고(필터에 쓰이는 값이라 끄면 전송이 안 될 수 있음) 바꾸지 않았다.
- **승인**: 사용자(2026-11-09, STEP 10 진행과 Langfuse 요청 7건). `config_version` 정의 승인. 공개 키가 SDK 기본 동작으로 metadata에 들어가는 것은 **허용**("기본값이라 바꾸지 않아도 된다"). 발표 화면·내보낸 JSON에 이 값이 보일 수 있다는 점은 유의한다.

### D-039 STEP 11 RAGAS Faithfulness 구현 (Judge는 별도 기록, 프롬프트 승인 대기)
- **방식**: RAGAS 라이브러리를 쓴다(과제 요구: `user_input`, `response`, `retrieved_contexts`를 넘겨 점수를 구함). 설치하면 메인 환경의 `openai`가 3.46.0에서 3.3.0으로 내려가므로(ragas 0.4.3 기준) **격리 환경**에서만 실행한다: `uv run --with ragas --with "langchain-community<0.4" python scripts/run_ragas.py …`. `langchain-community<0.4`가 필요한 이유는 최신 버전이 ragas가 쓰는 모듈(`chat_models.vertexai`)을 없애 `import ragas`가 실패하기 때문이다(0.4.3, 0.3.9, 0.3.1 모두 같음). 이 환경에서 ragas 0.4.3, openai 3.3.0, langchain-community 0.3.31이 쓰이고 **메인 환경과 `pyproject.toml`·`uv.lock`은 바뀌지 않았다**(확인함). 평가는 "저장된 결과를 읽는 별도 단계"라는 설계와도 맞다.
- **"공식 예제"**: 과제는 문서·코드 링크 없이 입력 이름 세 가지만 지정한다. RAGAS 공식 문서 페이지는 직접 열어 보지 못했다. 설치한 라이브러리의 `Faithfulness` 클래스 설명에 들어 있는 사용 예(`llm_factory(모델, client=AsyncOpenAI())`, `Faithfulness(llm=llm)`, `await metric.ascore(user_input=, response=, retrieved_contexts=[…])`)와 우리 호출 형태가 같다. 옛 방식(`SingleTurnSample`)은 0.4.3에서 곧 사라질 경로로 경고가 뜬다.
- **내부 동작(소스 확인)**: 1단계 LLM 호출이 답변을 주장 목록으로 쪼개고, 2단계가 주장마다 `reason`과 `verdict`(0/1)를 내며, 코드가 `맞은 수 ÷ 전체 수`로 점수를 계산한다. **주장이 하나도 없으면 `NaN`**(N/A 경로)이고 2단계는 부르지 않는다. 기본 프롬프트(영어, `ragas/metrics/collections/faithfulness/util.py`)는 답변의 모든 문장을 주장으로 만들어서, 되묻기 문장도 "근거 없는 주장"으로 판정되어 0.0점이 나왔고 "자료에서 확인되지 않는다"는 문장은 판정이 오락가락했다(F-DEV-13이 같은 입력에서 0.75와 1.0).
- **수정한 것(사용자 승인, 2026-11-09)**: **1단계 프롬프트만** `prompts/faithfulness_statement_v1.json`으로 교체한다(기본 지시문 첫 문단과 아인슈타인 예시는 원문 그대로, 지시문에 "되묻기·요청·자료에서 확인되지 않는다는 안내 문장은 주장으로 만들지 말고 사실 주장이 없으면 빈 목록을 돌려라"는 문단과 빈 목록 예시 1개 추가). 2단계(판정) 프롬프트는 라이브러리 기본값이다. 파일의 sha256과 버전을 결과에 기록한다. `Faithfulness`를 상속한 `DetailedFaithfulness`가 같은 2단계를 부르되 ragas가 버리던 주장별 `reason`·`verdict`를 결과에 저장한다. ragas의 비공개 메서드(`_create_statements`, `_create_verdicts`, `_compute_score`)를 쓰므로 **ragas 버전을 고정·기록**해야 한다(교차 확인으로 ragas 자체 점수와 우리 계산이 같은지도 저장한다).
- **기록 규칙(과제 기준)**: 사실 주장이 없으면 N/A(점수 없음, 0 아님), 평가 호출이 실패하면 평가 불가(0이나 1로 바꾸지 않음), 평균은 점수가 있는 답변만으로 내고 평균 옆에 대상 수·N/A·평가 불가 수를 적는다. **횟수: development는 1회, 공식 실행은 3회 평균**(각 회의 점수·이유 저장, 평균과 범위 기록)이다(사용자 결정).
- **gpt-6-luna 호환**: RAGAS가 `temperature=1.0`과 `max_completion_tokens`로 호출하고 모델이 받아들인다. **RAGAS 기본 출력 한도 1,024토큰은 부족했다**: 첫 채점에서 F-DEV-24의 평가가 `IncompleteOutputException`(추론 토큰 1,237개가 한도를 넘음)으로 평가 불가가 되었다. `llm_factory(max_tokens=4096)`가 `max_completion_tokens=4096`으로 전달됨을 확인하고 4096으로 올렸다.
- **development 결과 (저장된 dense 답변 30개, answer-v2, `results/faithfulness-dev-001/`과 `-002/`, 로컬 전용·Langfuse 미사용)**: 001(한도 1024): 점수 29개, 평가 불가 1개, N/A 0개, 평균 0.973. **002(한도 4096, 같은 입력을 다시 채점): 점수 30개, 평가 불가 0, N/A 0, 평균 0.995**(점수가 1.0이 아닌 답변은 F-DEV-26 하나, 0.857). 001은 평가 불가 사례의 기록으로 그대로 보존했다. 평가 한 번의 사용량: 30개 답변에 호출 60회(답변당 2회), 입력 약 86천·출력 약 20천 토큰(추론 약 8천 포함), 비용 약 $0.018(추정, 캐시 입력 단가 반영), 시간 약 1.5분(동시 6개).
- **N/A 경로**: 30개 중 N/A는 0개였다. 되묻기·보류 답변 8개도 문서 내용을 설명한 문장이 있어 점수가 나왔다(예: 되묻기 답변 F-DEV-19, 30은 "확인해 주세요" 문장이 주장에서 빠지고 사실 문장만으로 1.0). 요청 문장만 있는 답변이 N/A가 되는 경로는 가짜 LLM 단위 테스트로만 확인했고 실제 데이터에서는 아직 나온 적이 없다.
- **점수 변동(측정)**: 001과 002는 같은 입력인데 두 번 모두 점수가 있는 29개 중 **4개(14%)의 점수가 달랐다**(F-DEV-02 0.667→1.0, F-DEV-03 0.75→1.0, F-DEV-29 0.8→1.0, F-DEV-26 1.0→0.857, 차이 0.14~0.33). 원인은 판정만이 아니라 **1단계 주장 쪼개기 결과가 매번 다르다**는 점이다(주장 목록이 두 번 똑같은 답변은 29개 중 6개, 예: F-DEV-02는 주장 3개 대 6개). 그래서 공식 실행은 3회 평균으로 하고 회차 사이 범위를 함께 보고한다. development 1회 점수는 ±0.2 정도 흔들릴 수 있다고 보아야 한다.
- **사용 통계 전송(정정 포함)**: ragas는 기본으로 사용 통계를 외부(`t.explodinggradients.com`)로 보낸다. 이벤트 정의(소스 확인)상 내용은 익명 사용자 ID, ragas 버전, 사용 모델 이름(`gpt-6-luna`), 지표 이름, 호출 횟수 등이고 **질문·답변·문맥 텍스트는 포함되지 않는다.** 이를 시험을 시작하기 전에 확인하지 못해서 **지금까지의 시험과 첫 채점(001)에서는 전송이 켜져 있었다**(소량). 지금은 ragas를 import하기 전에 `RAGAS_DO_NOT_TRACK=true`를 설정해 끄고(002부터 manifest에 `ragas_telemetry: disabled`), 아무것도 전송하지 않음을 테스트로 확인한다. ragas가 PC에 익명 ID 파일(`%LOCALAPPDATA%\ragas\uuid.json`)을 만들었을 수 있으며 저장소 밖이다(삭제는 사용자 판단).
- **평가 호출의 Langfuse 기록과 서비스 기록의 구분(실제 확인, 2026-11-09)**: `src/evaluation/scoring_trace.py`의 `record_scoring`이 평가 호출을 서비스와 **별개의 Trace**(이름 `faithfulness-evaluation`, Generation, 태그 `run_type:scoring`, metadata에 평가 대상의 `request_id`·`trace_id`)로 기록한다(`run_ragas.py --langfuse`, 평가 Trace ID는 결과 파일에 저장). 서비스 `rag-query` 아래에 두지 않는다. STEP 10에서 남겨 둔 이 구분을 **사용자 승인으로 3건(Dense, Hybrid, Hybrid+Rerank 답변, 유닛 약 3개)을 실제 Langfuse에 기록해 읽어 확인했고 모두 통과했다**(`results/faithfulness-langfuse-check-001/langfuse_check.json`): 평가 Trace는 관측 1개(이름 `faithfulness-evaluation`, 유형 GENERATION, 부모 없음)이고 서비스 Trace와 다른 Trace, 모델명 `gpt-6-luna`, 평가 호출의 입력·출력 토큰이 결과 파일과 일치, metadata `run_type=scoring`·평가 대상 `request_id`·`trace_id`(서비스 Trace와 일치)·평가 도구·ragas 버전·프롬프트 버전, 출력에 점수·상태·주장별 이유, 입력에 정답 필드 없음, 비밀 값 없음, **서비스 Trace에는 평가 관측이 섞이지 않았고** 생성 호출은 모두 `answer-generation`이며 로컬 `llm_calls` 수와 같다. 다만 **태그 `run_type:scoring`은 Langfuse에서 읽어 확인하지 못했다**(SDK 5.2.0의 `api.trace`에 조회 메서드가 없고 관측 조회에는 태그가 없음. 코드에서 설정하는 것은 단위 테스트로 확인했고 metadata의 `run_type`은 읽어서 확인함). development 실험은 로컬에만 남기는 것이 기본이라(D-029) 이 3건 외에는 Langfuse로 보내지 않았다.
- **한계**: ① 점수 한 번은 변동이 크다(위). ② Faithfulness는 문맥에 근거했는지만 보고 **정확한지는 보지 않는다**(문맥이 틀렸거나 질문에 맞지 않는 값을 충실히 옮겨도 높다, 과제도 Judge와 함께 보라고 함). ③ 평가 모델이 서비스 생성 모델과 같다(자기 평가 편향 가능성, 06 §11.1 한계). ④ ragas 비공개 메서드 사용, ragas 버전 고정 필요. ⑤ 점수가 1.0에 몰려 있어(30개 중 26개) development 질문만으로는 방식을 가르기 어려울 수 있다.
- **아직 하지 않은 것**: LLM-as-a-Judge 4항목(프롬프트 문구를 만들기 전에 전문을 보여 드리고 승인받는다), 검색 평가를 공식 형식으로 합치기, 사람의 대표 사례 원문 대조(사용자 확인).
- **영향**: 02 §6.1, 06 §11.1, 08 §15 11.3, 09 §11(`faithfulness`, `evaluator`)
- **승인**: 사용자(2026-11-09): 1단계 프롬프트 문구와 새 예시, 횟수(development 1회·공식 3회 평균), 격리 환경 방식.

### D-040 STEP 11 LLM-as-a-Judge 4항목 구현 (프롬프트 `judge-v1`)
- **승인(사용자, 2026-11-09)**: `prompts/judge_v1.json`의 문안 전문, ① 첫 단락에서 '최종 문맥'(모델에 실제 전달한 검색 결과, 질문은 별도)을 한 이름으로 정의, ② **`evidence_groups`는 Judge 입력에서 제외**(등록 문서를 체크리스트로 쓰는 것을 막음. 과제도 모든 등록 문서 인용을 요구하지 않음), ③ **decision은 시스템 상태가 기대 동작과 다르면 무조건 0**, ④ **상태 비교는 코드가 먼저 처리**. 문안 확인 과정에서 제가 처음 쓴 "검색 자료만 보고 만들어졌다"는 표현이 부정확해(모델은 질문도 본다) 고쳤다. 승인 후 제가 정한 구현 세부는 아래에 따로 적었다.
- **무엇을 채점하나**: 사실(facts)·조건(conditions)·근거(evidence)·판단(decision). 항목마다 1 / 0 / 해당 없음 / 평가 불가와 짧은 이유. **모델은 1, 0, 해당 없음만 낸다. 평가 불가는 코드가 정한다**(호출 실패, 출력 형식 오류, 출력 잘림, 서비스 오류로 답변 없음).
- **Judge 입력**: 질문, 시스템 응답(상태·답변·이유), 인용(S번호, 파일, 페이지), **서비스 모델이 받은 형식 그대로의 최종 문맥**(`[S1] 본문`), 기대 동작(+API 상태 이름), 코드가 계산한 상태 비교(일치/불일치), 참고 답변, 필수 사실, 금지 주장, 채점 설명. 평가 파일의 정답 정보는 이 평가 단계에서만 쓰며 서비스 코드(`src/api`, `src/generation`, `src/retrieval`)는 평가 모듈을 가져오지 않는다(테스트로 확인).
- **코드 규칙(`src/evaluation/judge.py`)**: 서비스 상태≠기대 동작이면 decision 0을 코드가 기록하고 모델의 원래 decision 출력은 `model_output`에 보존한다. 서비스 `error` 응답은 모델을 부르지 않고 decision 0, 나머지 평가 불가(과제 97행). 평균은 점수가 있는 답변만으로 구하고 항목별 점수가 있는 수, 해당 없음, 평가 불가를 함께 적는다. 일반 평가(`run_type=general_eval`)가 아닌 기록(모의 오류 등)과 평가 파일에 없는 질문은 평균에서 빼고 개수만 센다.
- **승인된 문안과 달라진 점(모두 문구가 아닌 형식)**: ① 사용자 메시지의 `상태 비교: {status_match}` 줄 뒤에 초안에 붙였던 설명(`← 코드가 계산…`)은 사용자에게 보여 준 주석이라 파일에는 넣지 않았다. ② 출력 형식은 프롬프트 본문이 아니라 API의 JSON schema(strict)로 지정한다(본문은 "지정한 JSON 형식으로만 출력하세요"). ③ 프롬프트 JSON에 `may_use_reference_data: true` 필드를 두고, 기존 "모든 프롬프트에 정답 필드 이름이 없어야 한다"는 테스트를 이 필드가 있는 평가 프롬프트에 한해 예외로 바꿨다(서비스 답변 프롬프트와 다른 평가 프롬프트의 검사는 그대로, 이 필드는 평가 프롬프트에만 허용).
- **제가 정한 구현 세부(승인 문안 밖)**: 상태가 불일치하고 채점 호출도 실패한 경우 decision은 호출 결과와 무관한 규칙이라 0으로 기록하고 나머지 세 항목은 평가 불가로 둔다. 서비스 `error` 응답은 Faithfulness(대상 제외)와 달리 Judge 대상에 포함한다(과제 97행). 호출 설정은 gpt-6-luna, reasoning_effort low, 출력 한도 4096, timeout 120초, SDK 재시도 2회, **1회 채점**(회차별 변동은 아직 재지 않음). Langfuse 기록 시 참고 정답은 보내지 않는다(질문·답변·상태·점수·이유만).
- **development 결과 (저장된 dense 답변 30개, answer-v2, 로컬 전용·Langfuse 미사용, `results/judge-smoke-001/`(3개)와 `results/judge-dev-001/`(30개))**: 사실 19/30(0.63), 조건 29/30(0.97), 근거 29/30(0.97), 판단 26/30(0.87). 해당 없음 0, 평가 불가 0, 서비스 오류 0. 상태 불일치 4건(F-DEV-17, 22, 24, 25: 기대 answer인데 실제 abstained)은 코드가 판단 0으로 기록했다. 평가 호출 30회, 입력 74,600(캐시 7,416)·출력 11,858(추론 3,548) 토큰, 비용 약 $0.013(추정, 서비스 가격표 기준), 호출당 약 3~12초.
- **읽어 본 내용(AI가 읽음, 사람 확인 아님)**: 사실 0점 11건은 거의 모두 "필수 사실 누락" 이유다(예: F-DEV-14는 영업이익 감소·증가 사실을 빠뜨림, F-DEV-28은 회사 전체 매출 유지 설명을 빠뜨림). RAGAS Faithfulness(0.995)와 사실 충족률(0.63)이 크게 다른 것은 **두 지표가 다른 것을 재기 때문**이다(Faithfulness는 답변이 말한 것이 문맥에 근거하는지, 사실은 필요한 사실을 빠뜨리지 않았는지). 판단이 가능한 사례: F-DEV-19(되묻기)는 회사는 물었으나 기준기간·실적 항목을 묻지 않아 사실 0, F-DEV-13은 "자료에서 확인되지 않는다"는 문장이 인용으로 뒷받침되지 않는다며 근거 0(엄격한 해석일 수 있음). **Judge의 판정이 맞는지는 사람이 원문과 대조해 봐야 한다(미실시).**
- **한계·미확인**: ① 해당 없음 경로는 가짜 모델 단위 테스트로만 확인했고 실제 데이터에서는 한 번도 나오지 않았다(30개 모두 필수 사실이 있어 점수가 매겨짐). ② 같은 입력의 회차 간 변동은 재지 않았다(Faithfulness는 14%가 달랐다). 공식 실행 전에 회차 수를 정해야 한다. ③ 평가 모델이 서비스 생성 모델과 같다(자기 평가 편향 가능성). ④ 판단 항목은 상태 불일치를 규칙으로 0 처리하므로, 기대 동작이 애매한 질문은 평가 파일의 `expected_action`을 따른다. ⑤ Judge 호출의 Langfuse 기록(`judge-evaluation`, `run_type:scoring`)은 사용자 승인으로 3건을 실제로 보내 확인했다(D-042).
- **영향**: 02 §6.2, 06 §11.2, 08 §15 11.4, 09 §11(`judge`)
- **상태**: 구현·로컬 검증·Langfuse 기록 확인 완료. **사람의 원문 대조는 아직 하지 않았다**(작업지 D-042).

### D-041 development 30문항 세 방식 점검 (STEP 11 평가 도구 동작 확인, 공식 결과 아님)
- **목적과 범위**: 평가 도구(Judge, Faithfulness)가 세 방식의 답변에서 같은 방식으로 동작하는지 공식 실행 전에 보는 것이다(사용자 선택, 2026-11-09). **공식 비교 결과가 아니며 이 결과로 방식을 고르지 않는다**(공식 주결과는 additional, development는 튜닝용). 설정은 바꾸지 않았다(임베딩 e5, 공백 BM25, RRF k=60, 리랭커 bge-reranker-v2-m3, 최종 k=5, 프롬프트 answer-v2). 모두 로컬 전용·Langfuse 미사용(D-029). `scripts/run_dev_questions.py`에 `--rag-method`(기본 dense, 기존 동작·결과 동일)만 추가했다.
- **결과 폴더**: 서비스 실행 `dev-prompt-answer-v2-001`(dense, 기존), `dev-hybrid-answer-v2-001`, `dev-hybrid-rerank-answer-v2-001`. 채점 `judge-dev-001`·`judge-dev-hybrid-001`·`judge-dev-hybrid-rerank-001`, `faithfulness-dev-002`·`-hybrid-001`·`-hybrid-rerank-001`. 모두 서비스 오류 0건.
- **결과(30문항, 1회 채점)**:

| 지표 | dense | hybrid | hybrid_rerank |
|---|---|---|---|
| 응답 상태가 기대 동작과 일치 | 26 | 26 | 27 |
| Judge 사실 충족 | 19/30 (0.63) | 20/30 (0.67) | 20/30 (0.67) |
| Judge 조건 충족 | 29/30 (0.97) | 28/30 (0.93) | 29/30 (0.97) |
| Judge 근거 충족 | 29/30 (0.97) | 25/30 (0.83) | 28/30 (0.93) |
| Judge 판단 충족 | 26/30 (0.87) | 26/30 (0.87) | 27/30 (0.90) |
| Faithfulness 평균(점수가 있는 답변) | 0.995 (30) | 0.995 (29, N/A 1) | 0.985 (30) |
| 등록 근거 쪽(문서+페이지)이 최종 문맥에 있는 비율(임시 계산) | 0.46 | 0.41 | 0.51 |

- **읽을 때 주의**: 방식 간 차이는 한두 문항 수준이고(예: 사실 19→20→20), 질문이 30개이며 Judge·Faithfulness 모두 1회 채점이라 **순위를 말할 수 없다**. Judge의 회차 간 변동은 재지 않았다. 마지막 행은 `evidence_groups`의 (book_id, page)가 최종 문맥 청크에 있는지(쪽 단위 Recall)를 제가 계산한 값이다. 공식 Hit@k/Recall@k(문서 단위)는 D-042에 기록했고 같은 계산 경로의 쪽 단위 값과 일치한다.
- **관찰**: ① 세 방식 모두 등록 근거 쪽의 절반 정도만 최종 문맥에 들어 있다(0.41~0.51). 사실 0점의 대부분(10건 중 9~10건)이 근거 쪽 누락을 동반한다는 점은 방식과 무관하게 같다. 즉 k=5 문맥이 이 질문들에 부족한지가 다음 확인 대상이지만 **설정은 바꾸지 않았고 바꿀지는 사용자 결정 사항**이다. ② 상태 불일치는 기대 answer인데 보류·되묻기인 경우다(dense F-DEV-17, 22, 24, 25 / hybrid 17, 21, 22, 25 / rerank 17, 22, 25). F-DEV-24는 hybrid와 rerank에서만 답했고 F-DEV-21은 hybrid에서만 보류했다. ③ **Faithfulness N/A가 실제 데이터에서 처음 나왔다**: hybrid의 F-DEV-12 답변은 "확정할 수 없고 자료가 필요하다"는 안내뿐이라 사실 주장이 없다고 판단했다(제가 답변을 읽어 맞다고 봄, 사람 확인 아님). Judge의 해당 없음은 여전히 실제 데이터에서 나온 적이 없다.
- **평가 호출 사용량(분리 기록)**: Judge는 방식마다 30회·입력 약 75천·출력 약 12천 토큰, Faithfulness는 방식마다 약 60회·입력 약 85천·출력 약 19천 토큰.
- **한계**: 위 읽을 때 주의 외에, 평가 모델이 생성 모델과 같다(자기 평가 편향 가능성), Judge 판정이 맞는지는 사람이 원문과 대조하지 않았다.
- **영향**: 없음(설정·데이터 구조 변경 없음).

### D-042 STEP 11 마무리: 검색 지표(Hit@k·Recall@k) 공식 형식, Judge Langfuse 확인, 원문 대조 작업지
- **검색 지표(`src/evaluation/retrieval_metrics.py`의 `evaluate_record`·`aggregate_records`, `scripts/evaluate_retrieval.py`)**: 서비스 run 레코드의 검색 결과와 평가 파일의 `evidence_groups`로 계산한다(모델 호출·Langfuse 없음). 규칙은 과제 04_step6 그대로다: ① k는 답변 생성 직전에 선택한 상위 청크 수(기록된 `final_k`=5, 세 방식 같음) ② 참고 근거의 `book_id`를 모아 중복 제거한 목록 기준, 같은 문서의 청크가 여러 개여도 한 번만 셈 ③ 참고 목록이 비면 해당 없음(이유 기록) ④ 검색 결과가 비어도 참고 문서가 있으면 제외하지 않고 0 ⑤ 서비스 오류로 검색 기록이 없으면 평가 불가(0으로 바꾸지 않음) ⑥ 같은 참고 목록을 세 방식에 사용. 문맥 토큰 한도로 청크가 제외되면 선택된 k개와 최종 문맥을 구분해 최종 문맥 기준 지표를 따로 낸다(과제 65행). (book_id, page) 단위 지표는 참고용이다.
- **결과(development 30문항, `results/retrieval-eval-dev-001/`)**: 점수 28문항, 해당 없음 2문항(F-DEV-19, F-DEV-30: 기대 clarify이고 참고 근거 없음), 평가 불가 0. 문서 단위 Hit@5/Recall@5: dense 0.857/0.765, hybrid 0.929/0.753, hybrid_rerank 0.893/0.810. 쪽 단위 Hit/Recall: dense 0.571/0.461, hybrid 0.536/0.407, hybrid_rerank 0.643/0.507. **세 방식 모두 문맥 한도로 제외된 청크가 없어(제외 문항 0) 선택된 5개와 최종 문맥이 같다.** 이 값은 STEP 5·6에서 따로 계산한 검색 비교(`results/dev-retrieval-compare-002`)와 문서·쪽 단위 모두 정확히 일치한다(독립된 두 계산 경로로 교차 확인). 비교는 development 30문항이고 공식 결과가 아니다.
- **Judge 호출의 Langfuse 기록 확인(사용자 승인, 3건·유닛 약 3개, `results/judge-langfuse-check-001/`)**: 대표 3건(F-DEV-01 정상 답변, F-DEV-17 상태 불일치로 코드가 판단 0 처리, F-DEV-19 되묻기)을 `judge-evaluation`으로 보내 읽어 확인했고 **모든 확인이 통과했다**: 평가 Trace는 관측 1개(GENERATION, 부모 없음)이고 서비스 Trace와 다른 Trace, 모델명 gpt-6-luna, 입력·출력 토큰이 결과 파일과 일치, metadata `run_type=scoring`·평가 대상 `request_id`·`trace_id`·평가 도구·`prompt_version=judge-v1`·상태 비교, 출력의 4항목 점수·상태·이유가 결과 파일과 같음(불일치 건은 판단 0·코드 판정 기록), 입력에 참고 답변·필수 사실·금지 주장·채점 설명 문장과 필드 이름이 없음, 비밀 값 없음. **한계**: ① 태그 `run_type:scoring`은 Langfuse에서 읽어 확인하지 못했다(D-039와 같은 제약, 메타데이터의 `run_type`만 확인). ② 이 3건의 서비스 실행은 로컬이라 서비스 Trace가 Langfuse에 없다. "서비스 Trace에 평가 관측이 섞이지 않음"은 이 확인에서 다시 보지 못했고, 같은 `record_scoring`으로 D-039에서 확인한 결과에 의존한다(평가 Trace를 별도 Trace로 만들고 metadata로만 연결하는 구조). 이로써 STEP 10에서 남겨 둔 "평가 호출과 서비스 호출 구분"은 Faithfulness와 Judge 호출 모두 확인했다.
- **사람의 원문 대조 작업지**: `results/human-check-001/worksheet.md`(AI가 데이터에서 모은 자료, 확인 결과 칸은 비어 있음). 사례 10개(필수 5, 선택 5)이며 질문, 시스템 답변과 인용한 청크, 정답 근거(파일·쪽), 최종 문맥에 들어간 쪽, 필수 사실, Judge 4항목과 이유, Faithfulness 주장별 판정을 모았다. 선택 기준: Judge가 엄격했을 수 있는 0점(F-DEV-13, 14, 26, 19), 서비스가 보류한 것이 맞았는지(F-DEV-24, 17), 모두 1점인 사례의 오탐 점검(F-DEV-01, 27), 해당 없음 판정이 맞는지(F-DEV-12). **사람이 아직 대조하지 않았으므로 Judge·Faithfulness 판정이 맞다고 보고할 수 없다.**
- **영향**: 02 §5, 08 §15 11.1·11.5, 09 §11(`retrieval metrics`)
- **상태**: STEP 11 구현 완료. 남은 것은 사람의 원문 대조 결과를 받아 D-040·D-041에 반영하는 일이다.

### D-043 사람 대조 결과(Judge·Faithfulness)와 평가 기준 확정 (`judge-v2` 준비)
- **사람이 직접 확인한 것 (Judge, 2026-11-10)**: 사용자가 `results/human-check-001/worksheet.md`의 **필수 5사례**를 PDF와 대조했다. Judge 판정 20개(5사례 × 4항목) 중 **19개 동의, 1개 불일치**. 불일치는 사례 1(F-DEV-13 dense)의 **근거=0**이고 이유는 "최종 문맥에 실제로 그 내용이 없기 때문"이다. 사례 2~5(F-DEV-24 두 방식, F-DEV-12 hybrid, F-DEV-01 rerank)는 모두 동의. 표본이 5사례로 작고 선택 사례 6~10은 하지 않았다.
- **Judge v1 관찰 (AI가 90건 전체에서 집계)**: "확인되지 않는다/공개되지 않았다" 같은 부정 표현이 들어 있는 답변 37건 중 근거=1이 31건, 0이 6건이고 그중 이유가 부정 진술인 것이 4건(Dense F-DEV-13, Hybrid F-DEV-20, Rerank F-DEV-09, F-DEV-22)이었다. 같은 종류의 문장을 일관되지 않게 다룬다. 4건 모두 **최종 문맥에 그 내용이 실제로 없어서(허위 부정 0건)** 기준 A를 적용하면 근거=1이다(근거 충족률이 Dense 29→30/30, Hybrid 25→26/30, Rerank 28→30/30으로 바뀔 것으로 **추정**, Judge 재실행 값 아님). 또 Judge 입력의 최종 문맥에 쪽 번호가 없어서 "인용한 쪽이 맞는가"를 검증할 수 없는데도 감점한 사례가 있다(Hybrid F-DEV-01, F-DEV-26).
- **확정한 평가 기준 (사용자)**
  1. **근거 항목의 부정 진술(기준 A)**: 답변이 "검색 자료(제공된 자료)에서 확인되지 않는다/공개되지 않았다"고 했고 그 내용이 **최종 문맥에 실제로 없으면** 근거 감점 대상이 아니며 인용도 필요 없다. 문맥에 있는데 없다고 했으면(허위 부정) 근거=0. "PDF에 없다/세상에 없다"처럼 검색 자료가 아닌 것에 대한 단정은 면제되지 않는다. 문서가 "공개하지 않았다"고 명시한 문장을 인용한 경우는 일반 주장이다.
  2. **부정 진술(또는 확인 요청)만 있고 근거가 필요한 다른 주장이 없는 답변의 근거는 해당 없음(N/A)**(AI 제안을 사용자가 수락). Faithfulness가 사실 주장이 없으면 N/A인 것과 같은 원칙이다.
  3. **정답 파일(골든 데이터셋)은 사용자가 신뢰하는 전제**로 둔다. AI가 확인한 것은 근거 원문 166개(development 64, additional 62, expansion_validation 40)가 모두 해당 PDF 쪽의 추출 텍스트에 그대로 있다는 것이며, 필수 사실이 그 원문의 바른 요약인지는 검증하지 않았다.
- **Faithfulness 사람 확인 (2026-11-10)**: 기존 작업지에는 Faithfulness 칸이 없어서(AI의 설계 누락) 보충 작업지 `worksheet_faithfulness.md`(7사례)를 새로 만들었다. **사용자가 채팅으로 판단을 알렸고 AI가 작업지에 대신 기록**했다(사용자가 직접 칸을 채운 것이 아니며, 사례 4·6·7 외에는 세부 항목별 기록이 아니다).
  - 사례 1, 2, 3, 5: 문제 없음(구두).
  - **사례 7(F-DEV-12 dense)**: 문제 없음.
  - **사례 4(F-DEV-12 hybrid, AI 판정 N/A)**: **(1)번과 (2)번 문장을 모두 사실 주장으로 센다 → AI의 N/A 판정에 불일치.** (2)번 "상반기 실적과 6월 말 수주잔고만으로는 확정 매출을 산출할 수 없다"는 문서(S2 등)가 직접 설명하는 업무 규칙이고, (1)번 "검색 자료만으로 … 확정할 수 없다"도 문맥 S2의 "다음 분기의 매출 전망치를 공개하지 않았다"가 뒷받침한다는 판단이다. (1)번은 처음에 AI가 "검색 자료에 대한 진술이라 제외가 맞다"고 했던 것을 사용자가 바로잡았다.
  - **사례 6(F-DEV-26 dense, 0.857)**: 답변 "아니요."를 "…모두 돈을 받은 날의 매출로 설명할 수 없습니다"로 늘린 **재서술은 문제 없다**고 판단했고, 따라서 **이 주장은 0이 아니라 1**(뒷받침됨)로 본다 → **AI의 0 판정에 불일치(AI가 너무 엄격)**, 사람 기준 점수는 7/7 = 1.0. 처음에 AI가 "0 판정에 별도 이의가 없어 동의로 해석"했던 것은 사용자가 정정했다. AI는 재서술을 부풀림으로 보고 불일치를 추천했으나 사용자 판단을 따른다. 이 0은 **RAGAS 2단계(판정)**에서 나온 것이라 1단계 프롬프트를 바꿔도 해결되지 않는다. 같은 입력의 1회차(dev-001)에서는 이 주장이 만들어지지 않아 1.0이었다는 실행 간 변동도 사실로 남는다.
  - **사람도 사례 4, 6, 7은 판단하기 어려웠다.** 세 사례는 기준이 애매한 경계 사례이고, 과제 원문이 보류 답변의 "확인 불가" 문장 처리를 정하지 않았다는 점이 원인이다. 이 모호함 자체를 결과로 남긴다.
- **Faithfulness 프롬프트 결정 (갱신 2, 확정: ① v1 유지 + 한계 기록)**: 사용자의 판단(사례 4의 (1), (2)번 문장 모두 주장, 사례 6의 0은 과도함)과 v1 사이에 차이가 있다. 코드로 확인한 사실: **RAGAS 1단계(주장 쪼개기)는 질문과 답변만 받고 문맥을 보지 못하며**(`StatementGeneratorInput`: question, answer) 문맥은 2단계(판정, `NLIStatementInput`: context, statements)에서만 쓰인다. 그래서 "문서가 직접 그렇게 말하는 진술만 주장으로 센다"는 기준을 1단계 프롬프트로는 구현할 수 없다. 앞서 제안한 `faithfulness-statement-v2`(규칙 (c) 좁히기)는 이 한계 때문에 사례 4의 (1)번과 (2)번을 일관되게 구분한다는 보장이 없어 **보류**한다. 선택지: **① v1 유지 + 한계 기록(AI 권고)**, ② v2(1단계만 수정, 효과 불확실), ③ 1단계와 2단계(판정) 프롬프트를 모두 맞춤 설계해 부정 진술도 주장으로 세고 "문맥에 그 정보가 없고 모순도 없으면 뒷받침됨, 문맥에 있는데 없다고 하면 0, 문서가 직접 말하면 1"로 판정(Judge 기준 A와 같은 원리, RAGAS 기본 방식에서 더 벗어나고 승인·구현·재검증이 필요). 영향: Faithfulness N/A는 90건 중 1건이고 사례 6 같은 변동은 공식 실행의 3회 평균이 줄인다. 어느 쪽이든 **한계로 기록**한다. **사용자가 AI 권고 ①을 선택했다(2026-11-10 "지금 제안한대로 할게").** `faithfulness_statement_v1`은 그대로 유지하고 아래 한계를 최종 문서에 적는다: ⓐ RAGAS 1단계는 문맥을 볼 수 없어 "문서가 직접 말하는 확인 불가 진술"과 "단지 자료에 없다는 진술"을 구분하지 못하고 둘 다 주장에서 제외하므로 사례 4처럼 N/A가 될 수 있다(사람은 주장으로 센다) ⓑ 2단계 판정이 사례 6에서 사람보다 엄격했다(0 대 1) ⓒ 1단계 재서술은 실행마다 달라진다.
- **한계 (기록용)**: 사람 확인 표본이 Judge 5사례(판정 20개)로 작고, 기준 A는 development 관찰에서 나온 것이다. 그래서 **additional 공식 실행 결과를 보기 전에 `judge-v2`와 이 기준을 고정**해야 한다. 쪽 번호 입력 한계와 "필수 사실이 최종 문맥에 있었는가"를 사실 단위로 기록하지 않는 점(쪽 단위 어림값뿐)은 `judge-v2`에서 다룬다(승인 대기).
- **다음**: 승인된 `judge-v2`를 구현하고 development 90건을 새 폴더로 재채점해 v1과 비교했다(→ D-044).
- **영향**: 02 §6.2, 06 §11.2, 08 §15 11.4, D-039·D-040·D-041 해석
- **승인**: 사용자(2026-11-10): 기준 A, 부정 진술만 있는 답변의 근거 N/A, 골든 데이터셋 전제. Faithfulness 사례 4·6·7에 대한 사용자의 직접 판단(위). `judge-v2` 문안 승인(→ D-044), Faithfulness ① v1 유지.

### D-044 `judge-v2` 구현과 development 재채점 (v1 대비, 공식 결과 아님)
- **승인(사용자, 2026-11-10 "지금 제안한대로 할게")**: `judge-v2` 문안 전문 — 근거 항목의 부정 진술 규칙(기준 A)과 부정 진술만 있는 답변의 근거 N/A, 쪽 번호 출처 표, 필수 사실의 최종 문맥 포함 진단(점수에 영향 없음) — 과 `[근거 항목의 부정 진술 규칙]` 넷째 항목 수정안("확정할 수 없다" 포함), 그리고 Faithfulness ①(v1 유지). 승인된 문안을 글자 그대로 `prompts/judge_v2.json`으로 만들었고 v1과 비교해 system 25줄, user 6줄이 다르다(나머지 문장은 같음을 테스트로 확인).
- **구현**: `src/evaluation/judge.py`(출처 표 `format_source_table`, 필수 사실 번호 목록, 진단 해석 `parse_context_check`, 진단 집계, 프롬프트 버전에 맞는 출력 형식 `response_format_for`), `scripts/run_judge.py --prompt judge_v1|judge_v2`(기본 judge_v2). **judge-v1은 `--prompt judge_v1`로 그대로 재현할 수 있고 v1 결과 폴더(`judge-dev-*`)는 보존**했다. 진단 항목이 형식에 맞지 않으면 진단만 평가 불가로 두고 네 항목 점수는 유지한다. 단위 테스트 10개 추가, **전체 301개 통과(1개 건너뜀)**. 실행 중 발견한 결함 하나를 고쳤다: v1 입력에도 출처 표를 항상 만들다가 쪽 정보가 없는 기록에서 오류가 났다(v2일 때만 만들도록 수정).
- **결과 (development 90건, 방식당 30건, **1회 채점**, 로컬 전용, `results/judge-v2-dev-*-001/`)**

| 항목 | dense v1→v2 | hybrid v1→v2 | hybrid_rerank v1→v2 |
|---|---|---|---|
| 사실 | 19→18 /30 | 20→19 /30 | 20→19 /30 |
| 조건 | 29→30 | 28→30 | 29→29 |
| 근거 | 29→30 | 25→29 | 28→29 |
| 판단 | 26→26 | 26→26 | 27→27 |

  전체 360개 판정 중 **16개(4.4%)가 바뀌었다**: 근거 8(0→1 7건, 1→0 1건), 조건 5(0→1 4건, 1→0 1건), 사실 3(모두 1→0), 판단 0. 근거 0→1 7건 중 **6건은 부정 진술 규칙(기준 A)을 적용한 변화**(Dense F-DEV-13, Hybrid F-DEV-03·F-DEV-25·F-DEV-26, Rerank F-DEV-09·F-DEV-22, 모두 v2 이유에 "최종 문맥에 없어 감점 대상이 아니다"라고 적힘)이고, **1건(Hybrid F-DEV-01)이 출처 표로 인용 쪽을 검증할 수 있게 된 효과**(v2 이유에 출처 표의 쪽을 사용)이다. Hybrid F-DEV-26은 v1에서 출처 표 문제와 부정 진술이 함께 이유였으나 v2 이유는 부정 진술 규칙만 언급한다.
- **사람이 확인한 5사례(판정 20개)와 비교**: 사람 기준 일치 **v1 19/20 → v2 20/20**. **독립적인 검증이 아니다**: 기준 A는 이 사례를 보고 만든 것이고 사람 기준의 사례 1 근거=1이 v2 규칙이 겨냥한 바로 그 항목이다.
- **진단 결과 (필수 사실 63개/방식, LLM 판정)**: 문맥에 있음(yes) dense 47(75%) / hybrid 51(81%) / hybrid_rerank 52(83%), 일부(partial) 12 / 7 / 9, 없음(no) 4 / 5 / 2. **사실=0인 답변을 문맥으로 나누면** dense 12건 중 "필수 사실이 문맥에 모두 있었음" 2 / 일부만 10 / 전혀 없음 0, hybrid 11건 중 2 / 9 / 0, hybrid_rerank 11건 중 3 / 8 / 0이다. 즉 **사실=0의 대부분은 필요한 사실이 문맥에 일부만 들어온 경우(검색·문맥 구성 쪽이 유력)이고, 문맥에 다 있는데 모델이 놓친 경우(모델 쪽이 유력)는 방식당 2~3건**이다. 진단은 LLM 판정이라 **확정이 아니며**, 사람이 아는 두 사례와는 일치했다(F-DEV-13 dense: 필수 사실 1 partial / 2 yes — 장비사업 800억원이 문맥에 없고 620억원은 있었음. F-DEV-24 dense: 필수 사실 1 no / 2 yes).
- **정직하게 밝힐 문제**
  1. **원래 4건 중 1건은 여전히 0**: Hybrid F-DEV-20은 "모델별 영업이익은 공개되어 있지 않다"는 진술을 Judge가 "제공된 자료가 아닌 것(회사의 공개 여부)에 대한 단정"으로 보고 감점했다. 규칙 첫 항목은 "공개되지 않았다"를 면제 표현의 예로 들고 있어 Judge가 규칙을 일관되게 따르지 못한 사례이거나 규칙 문구의 모호함이다.
  2. **사실이 3건 모두 1→0**(Dense F-DEV-01, Hybrid F-DEV-18, Rerank F-DEV-21, 이유는 "확장형", "새 계약 금액 변경이 아님", "원료 투입 기준" 같은 필수 사실의 한 구절 누락). 사실 항목의 정의는 바뀌지 않았으므로 **채점 변동일 수도, 필수 사실에 번호를 붙인 입력 형식의 영향일 수도** 있어 둘을 구분할 수 없다. 방식마다 정확히 1건씩 내려가 비교 순위는 바뀌지 않았다.
  3. **Rerank F-DEV-18의 근거는 1→0**: 정정 전후 수치가 S4에 있는데 답변이 S4를 인용하지 않았다는 지적으로, 인용 일치 규칙의 엄격한 적용으로 볼 수 있다.
  4. **"부정 진술만 있는 답변의 근거 N/A" 규칙은 실제 데이터에서 한 번도 발동하지 않았다**(세 방식 모두 N/A 0건. 문서가 직접 설명한 문장으로 보아 1을 준 경우가 있었음).
  5. **Judge의 회차 간 변동은 여전히 측정하지 않았다.** 위 16건 중 사실 3건, 조건 5건, F-DEV-18 같은 규칙과 무관한 변화가 대략 9건(2~3%)이라 변동의 크기를 시사하지만 측정한 것은 아니다. 또한 방식 간 차이는 한두 문항 수준이라 **순위를 말할 수 없다.**
- **사용량**: 입력 +24%, 출력 +34%(진단과 출처 표 때문). 방식당 약 $0.013(추정).
- **결정 필요(STEP 12 시작 시)**: 공식 실행에서 **Judge를 몇 회 채점할지**(제안: 3회), 그리고 설정 동결. 이 기준과 프롬프트는 **additional 공식 실행 결과를 보기 전에** 고정했다.
- **영향**: 02 §6.2, 06 §11.2, 08 §15 11.4, 09 §11(`judge`)
- **상태**: 구현·로컬 검증 완료, 사용자 확인 대기.

---

## 4. 변경 이력

| 날짜 | 변경 | 내용 |
|---|---|---|
| 2026-11-06 | 최초 작성 | D-001~D-019 기록 |
| 2026-11-06 | 추가 | D-020 저장소 정책 기록, `.gitignore` 작성 |