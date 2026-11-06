# 04. ARCHITECTURE.md

## 1. 문서 목적

본 문서는 Financial Document Intelligence 프로젝트의 전체 시스템 구조와 구성요소별 책임, 데이터 흐름 및 모듈 간 연결 방식을 정의한다.

이 프로젝트는 금융·기업 관련 PDF 문서에서 질문에 필요한 근거를 검색하고, 검색된 근거를 바탕으로 답변하는 RAG 서비스다.

하나의 프로젝트 안에서 다음 항목을 선택하고 실행할 수 있는 통합 구조를 목표로 한다. 필수 구현은 Basic 단일 데이터셋 기준이며, 다른 데이터셋은 설정으로 확장한다([확장]).

- RAG 구성: Naive RAG (Dense) / Hybrid Retrieval / Hybrid + Reranking (`rag_method`: `dense` | `hybrid` | `hybrid_rerank`)
- 실험 설정: 청킹, 검색, 리랭킹, 프롬프트, 평가 및 운영 성능 관련 설정
- 데이터셋: Basic ([필수]), Intermediate / Advanced ([확장])

### 1.1 설계 원칙

1. 공통 기능은 최대한 재사용한다.
2. 데이터셋별로 필요한 문서 처리 방식과 메타데이터는 분리할 수 있다.
3. RAG 방식별 검색 전략은 교체할 수 있도록 설계한다.
4. 모든 RAG 방식의 실행 결과는 공통 결과 구조로 저장한다.
5. Basic의 필수 구현을 먼저 완성하고, 이후 Intermediate와 Advanced를 확장한다.
6. Advanced의 추가 기능이 기존 Basic 구현을 임의로 변경하거나 대체하지 않도록 한다.
7. 평가 및 실험 기능은 서비스 실행 기능과 구분한다.
8. 아직 선정하지 않은 기술은 실험과 검토를 거쳐 결정한다.

---

## 2. 전체 시스템 구성

### 2.1 주요 구성요소

| 구성요소 | 역할 | 구분 |
|---|---|---|
| FastAPI | 외부 요청 수신, API 검증, 응답 반환 | [필수] |
| LangChain | 문서 처리 및 LLM 기반 RAG 흐름 구성 | [필수] |
| GPT-6 Luna | 최종 답변 생성 | [필수] |
| Qdrant | 문서 청크의 벡터 저장 및 검색 (로컬 모드로 시작) | [필수] |
| Embedding Model | 문서 청크와 질문의 임베딩 생성 (로컬 모델) | [필수] |
| Sparse Retrieval | 키워드 기반 검색 (BM25) | [Hybrid 필수] |
| Reranker | 검색 후보의 관련성 재정렬 | [세 번째 RAG 방식 필수] |
| Langfuse (Cloud) | Trace, Span, 모델 입출력, 사용량 및 실행 시간 관측 | [필수] |
| Evaluation Module | 검색 및 답변 품질 평가 | [필수] |
| Experiment Runner | 설정별 실험 실행(스크립트)과 결과 저장·비교 | [필수] |
| Redis | 캐시 및 반복 요청 처리 효율 개선 | [확장] |
| Streamlit 또는 Gradio | 사용자 Q&A 및 실험 결과 확인 UI | [확장] |

※ Qdrant는 기본 벡터 DB로 확정했다. Docker가 없는 환경이므로 `qdrant-client` 로컬(내장) 모드로 시작하고, 설정값으로 서버 모드(Docker)로 전환할 수 있게 구현한다. 서버 모드 전환은 Advanced 또는 서비스 단계의 [확장]이다.

※ 임베딩 모델, Sparse Retrieval, Reranker의 초기값은 `06_MODEL_SELECTION.md`에서 정했다(공식 실행 전 동결). PDF 추출 라이브러리는 STEP 1에서 실제 PDF로 확인한 뒤 정한다.

### 2.2 전체 구조 개념도

```text
                    [사용자 / 평가 실행]
                            |
                            v
                       FastAPI
                            |
                            v
                  Request Validation
                            |
                            v
                   RAG Orchestrator
                   (rag_method 선택)
                            |
                            v
                     Retriever Layer
                            |
                +-----------+-----------+
                |           |           |
                v           v           v
              Dense       Sparse     Reranker
              Search      Search     (필요 시)
                |           |           |
                +-----------+-----------+
                            |
                            v
                  Context Construction
                  (S1, S2... 출처 ID 부여)
                            |
                            v
                   Answer Generation
                       (GPT-6 Luna)
                            |
                            v
              Output Validation / Citation Mapping
              (source_ids 검증 -> 문서명·페이지 변환)
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              Response             Langfuse
              to User              Tracing

              Evaluation / Experiment
              별도 실행 및 결과 분석
```

위 구조는 논리적인 구성도이며, 실제 구현에서는 각 구성요소를 별도 서버로 분리하지 않고 하나의 Python 애플리케이션 내부 모듈로 운영할 수 있다.

---

## 3. 프로젝트 실행 구조

### 3.1 통합 프로젝트 방식

Basic, Intermediate, Advanced를 각각 별도 프로젝트로 만들지 않는다. 필수 구현은 Basic 단일 데이터셋이며, 다른 데이터셋은 설정 파일의 필드로 확장한다([확장]).

예시:

```yaml
dataset:
  name: basic

rag:
  method: hybrid_rerank   # dense | hybrid | hybrid_rerank

model:
  generation: gpt-6-luna

retrieval:
  final_k: 5

experiment:
  name: basic_hybrid_rerank_v1
```

위 설정은 구조를 설명하기 위한 예시이며, 실제 설정 파일의 필드와 값은 구현 단계에서 확정한다(초기값은 `06_MODEL_SELECTION.md`).

실행 시 선택된 설정에 따라 검색 전략을 불러온다.

### 3.2 데이터셋별 실행 구조 [확장]

필수 구현은 Basic이다. 아래는 확장 시 방향이다.

| 데이터셋 | 처리 방향 | 구현 원칙 |
|---|---|---|
| Basic | 필수 기능 및 기본 RAG 파이프라인 | 최초 구현 기준 |
| Intermediate | 문서 규모와 특성에 맞는 처리 개선 | 공통 모듈 재사용 |
| Advanced | 대규모 문서 및 추가 검색 전략 검토 | 필요하면 별도 파이프라인 허용 |

데이터셋별로 다음 항목이 달라질 수 있다.

- PDF 추출 및 전처리 방식
- 청킹 전략
- 문서 및 청크 메타데이터
- 검색 필터
- 프롬프트 및 답변 처리 방식
- 인덱싱 및 검색 성능 설정

단, 데이터셋별 차이를 처리하기 위해 공통 시스템 전체를 복제하지 않는다.

### 3.3 RAG 구성별 실행 구조

| RAG 구성 | 검색 흐름 | 목적 |
|---|---|---|
| Naive RAG | Dense Retrieval → Context Construction | 기본 벡터 검색 |
| Hybrid Retrieval | Dense + Sparse Retrieval → 결과 결합 → Context Construction | 검색 범위 및 정확도 개선 |
| Hybrid + Reranking | Dense + Sparse Retrieval → 결과 결합 → Reranking → Context Construction | 최종 문맥의 관련성 개선 |

세 가지 방식은 동일한 공통 실행 시스템을 사용한다.

검색 방식에 따라 Retriever Layer의 구현만 교체하거나 조합한다.

최종 비교에서는 다음 조건을 통제한다.

- 동일한 PDF와 문서 추출 결과
- 동일한 청킹 및 임베딩 설정
- 동일한 생성 모델과 답변 프롬프트
- 동일한 생성 설정
- 동일한 최종 청크 수 k와 문맥 토큰 한도
- 동일한 timeout, 최대 호출 수, 재시도 횟수

검색 방식별 결합 방법과 Reranking 설정은 별도로 기록한다.

---

## 4. 주요 모듈과 책임

### 4.1 API Layer

**역할**

- FastAPI를 통한 사용자 및 평가 요청 수신 (공식 평가 실행도 이 API를 거친다. TestClient 허용)
- 요청 데이터 검증 (빈 질문, 지원하지 않는 `rag_method`)
- 서버에서 `request_id` 생성
- RAG 실행 모듈 호출
- 최종 응답 반환
- 오류 유형에 따른 HTTP 응답 처리 (`error` 상태는 API 코드가 판정)

**책임 범위**

API Layer는 검색 알고리즘이나 답변 생성 로직을 직접 구현하지 않는다.

요청을 검증하고 적절한 서비스 모듈에 전달하는 역할에 집중한다.

### 4.2 Orchestration Layer

**역할**

RAG 전체 실행 순서를 관리한다.

주요 흐름:

1. 요청 및 실행 설정 확인
2. RAG 구성(`rag_method`) 선택
3. 검색 실행
4. 최종 문맥 구성 및 출처 ID 부여
5. 답변 생성
6. 출력 검증 및 출처 변환, 답변 상태 처리
7. 결과 저장 및 Langfuse 관측 기록
8. API 응답 반환

각 단계의 세부 로직은 별도 모듈에 위임한다.

### 4.3 Document Processing Layer

**역할**

PDF 문서를 검색 가능한 형태로 변환한다.

주요 기능:

- PDF 파일 로드
- 텍스트 및 필요한 표 정보 추출
- 문서 메타데이터 생성
- 전처리
- 청킹
- 청크별 메타데이터 부여
- 임베딩 생성
- 벡터 DB 색인

데이터셋별 처리 차이는 이 Layer에서 관리한다.

예를 들어 Basic은 기본 청킹 방식을 사용하고, Advanced는 표 구조나 문서 구조를 고려한 별도 청킹 방식을 추가할 수 있다.

다만 Advanced 전용 처리 방식이 Basic의 기존 처리 방식을 변경하지 않도록 한다.

### 4.4 Dataset Configuration [확장 시 분리]

필수 구현(Basic 단일)에서는 별도 계층을 두지 않고 설정 파일의 필드로 관리한다(데이터셋 ID·버전, 문서 폴더, 청킹·색인 설정). 데이터셋이 여러 개가 되는 확장 단계에서 데이터셋 레지스트리와 데이터셋별 파이프라인으로 분리한다.

데이터셋 간 문서가 검색 대상에 섞이지 않도록 컬렉션 이름에 데이터셋과 버전을 포함한다(`07_VERSIONING.md`).

### 4.5 Retrieval Layer

**역할**

사용자 질문과 관련된 문서 청크를 검색한다.

구성:

- Dense Retriever
- Sparse Retriever
- Hybrid Retriever
- Reranker
- 검색 결과 결합 및 정렬

Naive RAG는 Dense Retriever를 사용한다.

Hybrid Retrieval은 Dense와 Sparse 결과를 결합한다.

Hybrid + Reranking은 결합된 후보를 Reranker로 재정렬한다.

검색 방식별 상세 설정은 RAG Configuration에서 관리한다.

### 4.6 Context Construction Layer

**역할**

검색 결과 중 실제 답변 생성에 사용할 문맥을 구성한다.

주요 기능:

- 검색 결과 정리 및 중복 처리
- 문서 및 페이지 정보 유지
- 최종 청크 수 제한
- 문맥 토큰 한도 관리 (줄여도 대상, 조건, 단위, 정정 내용, 출처는 남긴다)
- 청크마다 이번 요청에서 사용할 출처 ID(`S1`, `S2`…) 부여
- 출처 ID와 `book_id`, `file_name`, `page`의 대응표를 코드에서 보관
- 모델에 전달할 최종 문맥 생성 및 실제 전달 문맥 기록

중요한 원칙은 검색 결과와 실제 모델에 전달한 최종 문맥을 구분하여 기록하는 것이다.

평가 시에는 검색 결과 전체가 아니라 실제 답변 생성에 사용한 최종 문맥을 기준으로 Faithfulness와 Judge 평가를 수행한다.

### 4.7 Answer Generation Layer

**역할**

GPT-6 Luna를 이용해 최종 답변을 생성한다.

주요 기능:

- 사용자 질문과 최종 문맥 전달 (서비스 지시, 질문, 검색 자료를 프롬프트에서 구분)
- 답변 프롬프트 적용
- 답변 생성: LLM은 `status`, `answer`, `reason`, `source_ids`를 가진 JSON 하나를 반환한다.
- 코드에서 출력 검증: 상태 값, 필수 필드, `source_ids`가 이번 요청의 최종 문맥에 있는 ID인지 확인한다.
- 컨텍스트에 없는 ID 또는 형식 오류는 정상 답변으로 통과시키지 않고 출력 오류로 처리하거나 정한 횟수 안에서 다시 생성한다.
- 검증된 `source_ids`를 코드가 보관한 대응표로 문서명·페이지에 변환한다. 문서명과 페이지를 LLM이 만들지 않는다.

답변 상태는 다음과 같이 구분한다. 상태 판단(answered / needs_clarification / abstained)은 LLM이 검색 문맥과 프롬프트를 읽고 하며, 별도 판단 규칙 코드나 충돌 탐지 시스템은 만들지 않는다.

| 내부 기대 행동 | API 응답 상태 | 의미 |
|---|---|---|
| answer | answered | 근거를 바탕으로 답변 |
| clarify | needs_clarification | 필수 조건이 부족하여 추가 확인 |
| abstain | abstained | 근거가 부족하거나 확인할 수 없어 보류 |
| (없음) | error | 시스템 오류. LLM이 아니라 API 코드가 판정 |

부분적으로 답할 수 있는 질문은 확인 가능한 내용은 답하고, 확인할 수 없는 부분은 이유와 함께 구분하여 보류한다(필수 기본 동작).

### 4.8 Evaluation Layer

**역할**

검색 및 답변 품질을 평가한다.

주요 기능:

- Hit@k
- Recall@k
- RAGAS Faithfulness
- LLM-as-a-Judge
- 평가 결과 저장 및 비교

평가 모듈은 실제 RAG 실행 결과를 입력으로 사용한다.

정답, 정답 근거 및 expected_action은 평가 단계에서만 사용한다. 검색이나 답변 생성 단계에 전달하지 않는다.

### 4.9 Experiment Runner

**역할**

설정별 실험을 실행하고 결과를 비교한다. 서비스 모듈이 아니라 실행 스크립트(`scripts/`)로 구현한다. 공식 평가 실행은 `/rag/query` API를 거쳐 응답시간과 Langfuse Trace를 서비스 요청과 동일하게 측정한다.

주요 기능:

- 데이터셋 및 RAG 구성 선택
- 실험 설정 로드
- 동일 질문 반복 실행
- 설정별 결과 수집
- 결과 비교 및 기록
- 실험 버전 관리

공식 비교 실험과 개인 확장 실험을 구분한다.

확장 실험에서 변경한 설정이 공식 비교 설정을 덮어쓰지 않도록 별도 실험 설정과 버전을 사용한다.

### 4.10 Observability Layer

**역할**

Langfuse를 통해 실행 과정을 추적하고 운영 지표를 수집한다.

주요 기록 대상:

- 질문 및 요청 ID (`request_id`, `trace_id`, 평가 시 `question_id`)
- `run_type`(general_eval / security / mock_error / scoring) 및 RAG 구성
- 검색 결과
- 실제 모델 입력 및 출력
- 최종 답변과 상태
- 인용 정보
- 각 단계의 실행 시간
- 모델 사용량 및 토큰
- 오류 및 재시도 정보

하나의 요청에서 발생한 검색, 문맥 구성, 모델 호출 및 결과 반환을 동일한 Trace 아래 연결한다.

세부 Trace와 Span 구조는 `05_OBSERVABILITY_PLAN.md`에서 정의한다.

---

## 5. 데이터 흐름

### 5.1 PDF 색인 흐름 (Offline Pipeline)

```text
PDF Files
   |
   v
PDF Loading
   |
   v
Text / Table Extraction
   |
   v
Preprocessing
   |
   v
Metadata Assignment
   |
   v
Chunking
   |
   v
Embedding Generation
   |
   v
Qdrant Indexing
   |
   v
Index Ready
```

색인 과정은 사용자 질문에 답하는 실시간 RAG 요청과 분리한다.

문서 추가나 수정이 발생하면 해당 문서의 추출, 청킹, 임베딩 및 색인 과정을 다시 수행할 수 있어야 한다.

문서 및 인덱스의 버전 정보를 관리하여 어떤 문서 상태로 실험했는지 확인할 수 있도록 한다.

### 5.2 일반 질문 처리 흐름 (Online Pipeline)

```text
User Question
   |
   v
FastAPI Request
   |
   v
Request Validation
   |
   v
RAG Configuration (rag_method)
   |
   v
Retrieval
   |
   v
Context Construction (S1, S2... 부여)
   |
   v
Answer Generation (GPT-6 Luna)
   |
   v
Output Validation / Citation Mapping
   |
   v
Response
```

검색과 답변 생성 과정은 Langfuse Trace로 연결한다. 질의 재작성 등 Query Processing 단계는 [확장]이다.

평가 질문으로 실행할 경우 `question_id`를 함께 전달하여 해당 질문의 검색 결과, 답변, 인용 및 평가 결과를 연결한다. `question_id`는 기록용이며 검색과 답변 생성에 사용하지 않는다.

### 5.3 평가 흐름

```text
Evaluation Question
   |
   v
RAG Service Execution
   |
   v
Retrieval Result + Final Context
   |
   v
Generated Answer + Citation + Status
   |
   v
Evaluation Module
   |
   +----------------------+
   |                      |
   v                      v
Search Evaluation     Answer Evaluation
   |                      |
   v                      v
Hit@k / Recall@k      Faithfulness / Judge
   |                      |
   +-----------+----------+
               |
               v
       Evaluation Results
               |
               v
       Result Comparison
```

검색 평가와 답변 평가는 동일한 RAG 실행 결과를 기준으로 수행한다.

평가 과정에서 정답과 채점 기준은 평가 단계의 별도 입력으로 전달한다. 평가 실행은 `/rag/query` API를 통해 만든 결과를 입력으로 하며, 평가 LLM 호출은 `run_type = scoring`으로 구분한다.

### 5.4 보안 비교 흐름

보안 비교는 세 가지 RAG 방식 전체에 반복하지 않는다.

선택한 RAG 구성 하나에서 업무 질문 두 개를 검색하고, 각 질문에 대해 기본 프롬프트와 방어 프롬프트를 각각 적용한다.

```text
Security Question
       |
       v
    Retrieval
       |
       v
  Final Context
       |
       +----------------------+
       |                      |
       v                      v
 Basic Prompt           Defensive Prompt
       |                      |
       v                      v
 GPT-6 Luna             GPT-6 Luna
       |                      |
       v                      v
 Basic Answer           Defensive Answer
       |                      |
       +-----------+----------+
                   |
                   v
             Security Analysis
```

두 프롬프트의 비교에서는 검색 결과와 최종 문맥을 동일하게 유지한다. 저장한 최종 문맥을 두 실행에 그대로 넣으며 `attack_text`를 문맥에 따로 붙이지 않는다.

보안 질문, 공격 문장 및 보안 평가용 정답은 검색 단계에 전달하지 않는다. 보안 실행은 `run_type = security`로 기록하며 일반 평가의 운영 지표와 분리한다.

### 5.5 개인 확장 실험 흐름

필수 비교가 완료된 후에는 실험 설정을 변경하여 다양한 기법을 시험한다.

예시:

- 청킹 전략 변경
- Dense / Sparse 검색 비율 변경
- Query Expansion
- Reranking 후보 수 변경
- Context Token 조정
- 프롬프트 개선
- Redis 캐시 적용
- 추가 평가 지표 적용

확장 실험은 기존 필수 결과와 구분하여 저장한다.

---

## 6. API 구조

API는 RAG 실행을 중심으로 구성한다. 초기에는 필요한 API만 구현한다. 요청·응답 필드와 오류 계약은 `09_DATA_SCHEMA.md`를 따른다.

### 6.1 필수 API

| API | Method | 역할 | 구분 |
|---|---|---|---|
| `/health` | GET | 서비스 응답 확인 | [필수] |
| `/rag/query` | POST | 사용자 질문에 대한 RAG 실행 | [필수] |

### 6.2 확장 API 후보

| API | Method | 역할 | 구분 |
|---|---|---|---|
| `/datasets` | GET | 사용 가능한 데이터셋 조회 | [확장] |
| `/rag/configs` | GET | 사용 가능한 RAG 구성 조회 | [확장] |
| `/experiments` | GET | 실험 목록 조회 | [확장] |
| `/experiments/{experiment_id}` | GET | 실험 결과 조회 | [확장] |
| `/evaluations` | GET | 평가 결과 조회 | [확장] |

### 6.3 RAG 요청

```json
{
  "question": "해온솔루션의 상반기 매출과 영업이익이 모두 증가했는가?",
  "rag_method": "hybrid",
  "question_id": "F-DEV-01"
}
```

- `question`: 사용자 질문 (필수, 빈 문자열 불가)
- `rag_method`: `dense` | `hybrid` | `hybrid_rerank` (필수)
- `question_id`: 평가 실행 시 로깅·결과 연결용 선택 필드. 검색과 답변 생성에는 영향을 주지 않는다.
- 요청 ID(`request_id`)는 서버가 생성한다.
- 데이터셋은 요청 파라미터가 아니라 서버 설정으로 정한다(Basic 단일). 데이터셋 선택은 [확장]이다.

### 6.4 응답

API는 다음 정보를 반환한다.

- `request_id`: 서버가 생성한 요청 ID
- `status`: `answered` | `needs_clarification` | `abstained` | `error`
- `answer`: 답변 또는 확인할 질문
- `reason`: 판단 이유 또는 오류 사유 코드
- `citations`: 코드가 변환한 출처 목록 (`source_id`, `book_id`, `file_name`, `page`)

문서명과 페이지는 LLM이 아니라 코드가 보관한 대응표에서 변환한다.

### 6.5 오류 처리

오류는 원인에 따라 구분한다. 추가 확인과 답변 보류는 시스템 오류가 아니다.

| 오류 유형 | 예시 | HTTP | reason |
|---|---|---|---|
| 입력 검증 오류 | 빈 질문, 지원하지 않는 `rag_method` | 422 | `invalid_request` |
| 모델 호출 오류 | 모델 API 호출 실패 | 502 | `model_error` |
| 모델 시간 초과 | timeout·재시도 한도 도달 | 504 | `model_timeout` |
| 출력 검증 오류 | 잘못된 JSON, 컨텍스트에 없는 출처 ID | 502 | `output_validation_error` |
| 검색 오류 | 검색 저장소 연결 실패 | 500 | `retrieval_error` |
| 내부 오류 | 예상하지 못한 예외 | 500 | `internal_error` |

- 모든 오류 응답은 `status: "error"`와 `request_id`를 포함하며 응답 형식은 동일하다.
- 평가 LLM 호출 실패는 서비스 오류로 합산하지 않고 평가 불가로 기록한다.
- 일반 평가의 오류율에는 시스템 오류(`invalid_request` 제외)만 포함한다.
- 내부 비밀 값은 응답과 로그에 노출하지 않는다.
- Timeout과 재시도 값은 운영 지표 측정 전에 확정한다(§13.4).

---
## 7. 저장소 및 데이터 관리 구조

### 7.1 주요 저장 대상

| 저장 대상 | 목적 | 관리 위치 |
|---|---|---|
| 원본 PDF | 문서 원본 보존 | 파일 저장소 |
| 추출 문서 | PDF 처리 결과 | 파일 또는 데이터 저장소 |
| 청크 및 메타데이터 | 검색 단위 관리 | 인덱싱 데이터 |
| 벡터 임베딩 | Dense Retrieval | Qdrant |
| 실험 설정 | 실행 조건 재현 | 설정 파일 |
| 질문 및 정답 | 평가 실행 및 채점 | Evaluation Data |
| 실행 결과 | 답변 및 검색 결과 보존 | 결과 저장소 |
| 평가 결과 | 품질 평가 기록 | 결과 저장소 |
| 관측 기록 | 실행 추적 및 운영 지표 | Langfuse |

실험 결과는 파일 기반 JSON/JSONL로 저장한다(`09_DATA_SCHEMA.md`). 관계형 DB는 [확장]이다.

`request_id`, `question_id`, `experiment_id`를 기준으로 검색 결과, 답변, 평가 결과 및 Langfuse 기록을 연결할 수 있어야 한다. 비교표의 기준 데이터는 결과 JSON이며 Langfuse는 질문별 추적·시연용이다.

### 7.2 Qdrant 역할

Qdrant는 청크 임베딩을 저장하고 Dense Retrieval을 수행하는 기본 벡터 DB로 사용한다. 로컬(내장) 모드로 시작한다.

컬렉션 이름에 데이터셋, 청킹 버전, 임베딩 버전을 포함해 버전이 섞이지 않게 하고 기존 컬렉션은 삭제하지 않는다(`07_VERSIONING.md`).

BM25는 같은 청크 목록(`chunks.jsonl`)으로 메모리에서 색인한다(`06_MODEL_SELECTION.md`). Payload 필드와 인덱스 설정은 구현 단계에서 정한다.

### 7.3 Redis 역할

Redis는 초기 필수 구성에 포함하지 않는다.

향후 다음과 같은 확장 실험에 사용할 수 있다.

- Exact Cache
- Retrieval Cache
- Semantic Cache
- Embedding Cache
- 반복 요청 처리 및 비동기 작업 관련 기능

초기 구조에서는 Qdrant가 벡터 검색을 담당하고 Redis가 선택적 캐시를 담당하도록 역할을 분리한다.

Redis 도입 여부는 성능 개선 효과와 운영 복잡성을 비교한 후 결정한다.

---

## 8. Langfuse 연동 구조

Langfuse는 서비스 요청과 실험 실행을 관측하는 공통 시스템으로 사용한다.

### 8.1 Trace 구성 개념

하나의 사용자 요청을 하나의 Trace로 관리한다.

그 안에서 주요 처리 단계를 Span 또는 적절한 Observation으로 연결한다.

```text
Trace: rag-query
 |
 +-- request-validation
 |
 +-- retrieval
 |
 +-- context-construction
 |
 +-- answer-generation
 |
 +-- citation-and-status
 |
 +-- response
```

실제 Observation의 유형과 세부 명칭은 OBSERVABILITY_PLAN에서 확정한다.

### 8.2 필수 기록 정보

- Request ID, Trace ID
- `question_id` (평가 실행 시)
- `run_type` (general_eval / security / mock_error / scoring)
- Dataset 및 Dataset Version
- RAG Method 및 설정 버전
- Prompt Version
- 질문
- 검색 결과
- 실제 모델 입력
- 실제 모델 출력
- 답변 상태
- 인용 정보
- 실행 시간
- Input / Output Token
- 오류 및 재시도 정보

평가 LLM의 호출 기록은 서비스 답변 생성 호출과 구분한다.

API Key 및 비밀 정보는 Trace나 일반 로그에 기록하지 않는다.

---

## 9. 모듈 간 의존성 및 재사용 원칙

### 9.1 공통 모듈

다음 기능은 데이터셋이나 RAG 방식에 관계없이 공통으로 사용할 수 있도록 설계한다.

- FastAPI Request / Response 처리
- 공통 Orchestrator
- 모델 호출 인터페이스
- Langfuse 연동
- 결과 저장 및 조회
- 평가 실행
- 실험 실행 및 결과 비교
- 공통 오류 처리

### 9.2 데이터셋별 교체 가능 모듈 [확장 시 분리]

필수 구현(Basic)에서는 `src/ingest/`의 PDF 추출, 청킹, 색인 모듈로 둔다. 다른 데이터셋을 추가할 때 다음 기능을 데이터셋별로 교체할 수 있게 분리한다.

- PDF Loader, Parser, Preprocessor
- Chunker, Metadata Builder
- Dataset Configuration, Indexing Pipeline

### 9.3 RAG 방식별 교체 가능 모듈

- Dense Retriever
- Sparse Retriever
- Hybrid Retriever
- Reranker
- Context Builder 설정

### 9.4 의존성 관리 원칙

1. API Layer가 개별 검색 구현에 직접 의존하지 않도록 한다.
2. Orchestrator가 구체적인 검색 알고리즘을 직접 구현하지 않도록 한다.
3. 데이터셋 설정과 RAG 설정을 분리한다.
4. 검색·색인·문맥 구성·답변 생성 모듈이 정답, 근거, `expected_action` 등 평가 데이터에 접근하지 않도록 한다. 평가 모듈만 답변 생성 후 이 데이터를 사용한다.
5. 공통 결과 구조를 유지하여 서로 다른 RAG 방식의 결과를 비교할 수 있도록 한다.
6. 새로운 기능을 추가할 때 기존 기능의 동작이 바뀌지 않는지 회귀 테스트를 수행한다.

---

## 10. 프로젝트 디렉터리 구조

아래는 현재 저장소 구조를 기준으로 한 목표 구조이다. 디렉터리와 파일은 해당 STEP에서 필요할 때 만든다. 빈 모듈이나 불필요한 추상화는 만들지 않는다.

```text
financial-rag-qa/
├── CLAUDE.md
├── README.md
├── DECISIONS.md              # 승인된 결정과 이유
├── TASKS.md                  # 작업 진행 관리
├── .env.example              # 변수 이름만 (.env는 Git 제외)
├── .gitignore
├── pyproject.toml            # uv 기반 의존성
│
├── project_docs/             # 설계 문서 (00_assignment 원본은 수정하지 않음)
│
├── data/
│   ├── finance_basic/        # 제공 원본: documents/*.pdf, evaluation/*.json (수정 금지)
│   └── processed/            # 가공 결과: pages.jsonl, chunks.jsonl, 색인 매니페스트
│
├── configs/                  # 실행 설정 (공식 실험 설정 파일 등, 필요한 것부터)
│
├── src/
│   ├── api/                  # FastAPI 라우트, Pydantic 스키마
│   ├── pipeline/             # 질문 처리 순서, 최종 문맥 구성, 답변 생성, 출처 매핑, 프롬프트
│   ├── ingest/               # PDF 추출, 청킹, 임베딩·색인
│   ├── retrieval/            # dense, sparse(BM25), hybrid(RRF), rerank
│   ├── evaluation/           # Hit@k·Recall@k, Faithfulness, Judge
│   ├── observability/        # Langfuse 연동
│   └── core/                 # 설정 로딩, 오류 정의
│
├── scripts/                  # ingest, run_experiment, run_evaluation 실행 스크립트
├── tests/                    # unit / integration / regression
└── results/                  # {experiment_id}/ manifest.json, runs.jsonl, evaluations.jsonl, summary.json
```

저장 구조와 필드는 `09_DATA_SCHEMA.md`를 따른다.

주의사항:

- `data/finance_basic/`는 제공 원본이므로 수정하지 않는다. 검색 대상은 `documents/*.pdf`뿐이며 `evaluation/`은 색인하지 않는다.
- `.env`와 API Key는 Git에 업로드하지 않는다. 벡터 DB 데이터, 모델 가중치, 캐시도 Git에서 제외한다.
- `results/`는 기존 결과를 덮어쓰지 않는다. 저장소에 포함할지 여부는 크기를 확인한 뒤 결정한다.
- 확장 기능(Redis, UI, 다른 데이터셋)은 기존 구조를 바꾸지 않고 별도 모듈·설정으로 추가한다.

---
## 11. UI 및 사용자 인터페이스

### 11.1 초기 구현

초기에는 API 중심으로 개발한다.

UI는 필수 구현이 완료된 뒤 추가한다.

### 11.2 사용자 Q&A 화면 [확장]

Streamlit 또는 Gradio를 후보로 한다.

주요 기능:

- 데이터셋 선택
- RAG 방식 선택
- 질문 입력
- 답변 표시
- 답변 상태 표시
- 출처 문서명 및 페이지 표시
- 추가 확인 또는 보류 이유 표시

### 11.3 실험 결과 공유 화면 [확장]

팀원이나 사용자가 실험 결과를 확인할 수 있는 화면을 추가할 수 있다.

주요 기능:

- 데이터셋 및 RAG 방식별 결과 조회
- 질문별 결과 비교
- 검색 및 답변 평가 지표 확인
- 응답시간 및 토큰 비교
- 대표 실패 사례 확인
- 실험 설정 및 버전 확인

별도 운영 관리 시스템을 만드는 것은 목표가 아니다. 필요한 경우 간단한 Streamlit 화면으로 구현한다.

---

## 12. 확장 및 변경 관리

### 12.1 Basic 구현 보존

Basic에서 완성한 필수 기능은 Intermediate와 Advanced 확장의 기반으로 유지한다.

새로운 데이터셋이나 검색 전략을 추가할 때 기존 Basic 실행 결과가 달라지지 않도록 한다.

### 12.2 확장 기능 추가 방식

확장 기능은 다음 원칙에 따라 추가한다.

- 기존 필수 설정을 직접 수정하지 않고 별도 설정을 추가한다.
- 실험 이름과 버전을 구분한다.
- 기존 결과를 덮어쓰지 않는다.
- 새로운 모듈이 필요한 경우 기존 공통 모듈과 책임을 구분한다.
- 기존 기능에 영향을 주는 변경은 회귀 테스트를 수행한다.

### 12.3 설계 변경 승인

다음 항목은 사용자 승인 없이 변경하지 않는다.

- 핵심 기술 스택
- 전체 아키텍처
- 모듈의 책임과 데이터 흐름
- API Request / Response 구조
- 데이터 스키마
- 평가 기준
- 필수 요구사항
- Basic 필수 기능의 범위

설계 변경이 필요하면 다음 내용을 먼저 보고한다.

1. 현재 설계에서 발생한 문제
2. 변경이 필요한 이유
3. 기존 설계를 유지하는 대안
4. 변경 대안과 장단점
5. 영향받는 문서 및 모듈
6. 테스트 및 데이터 호환성 영향

사용자가 승인하기 전에는 변경안을 적용하지 않는다.

---

## 13. 서버 안정성 및 요청 처리 테스트

### 13.1 목적

FastAPI 서버와 RAG 시스템이 정상 요청, 잘못된 요청, 근거 부족, 외부 서비스 오류를 서로 구분해 예측 가능하게 처리하는지 확인한다.

세부 테스트 케이스와 실행 방법은 `08_DEVELOPMENT_PLAN.md`에서 정의한다.

### 13.2 Health Check [필수]

`GET /health`는 서비스가 응답하는지 확인하는 단일 엔드포인트로 둔다. Liveness와 Readiness의 분리는 [확장]이다(§13.7).

### 13.3 API 요청 처리 테스트 [필수]

다음 상황에서 API가 예상한 상태와 응답을 반환하는지 확인한다. HTTP 상태와 `reason` 값은 `09_DATA_SCHEMA.md`의 API 계약을 따른다.

- 정상적인 질문 요청 (`answered`)
- 필요한 조건이 빠진 질문 (`needs_clarification`)
- 근거가 부족한 질문 (`abstained`)
- 빈 질문, 지원하지 않는 `rag_method` (422, `invalid_request`)
- 모델 호출 실패·시간 초과를 모의한 오류 (`error`, 502/504)
- LLM 출력 형식 오류 또는 컨텍스트에 없는 출처 ID (`error`, `output_validation_error`, 또는 정한 횟수 안의 재생성)
- 검색 저장소 오류 (`error`, `retrieval_error`)
- Langfuse 연결 실패 시에도 답변과 결과 저장이 유지되는지
- 내부 처리 중 예외 발생 (`error`, `internal_error`)

오류가 발생하더라도 오류 유형과 원인을 기록하고, 적절한 HTTP 상태 코드와 응답을 반환해야 한다. 추가 확인과 답변 보류는 시스템 오류가 아니다. 모의 오류와 빈 질문 테스트는 일반 평가의 성능·오류율 집계에서 제외한다(`run_type = mock_error`).

### 13.4 Timeout 및 재시도 정책 [필수]

외부 서비스 응답 지연이나 일시적인 연결 오류에 대비하여 Timeout과 Retry Policy를 정의한다.

대상:

- 검색 저장소(Qdrant) 호출
- 생성 모델(GPT) 호출
- 임베딩·Reranker 호출 (API 사용 시. 로컬 모델은 해당 없음)
- Langfuse 기록

기본 원칙:

- 무한정 대기하지 않는다.
- 재시도 횟수에 상한을 둔다.
- 재시도 가능한 오류와 재시도해도 해결되지 않는 오류를 구분한다.
- 재시도 횟수와 대기 간격을 기록한다.
- 재시도로 인해 전체 응답시간과 토큰 사용량이 증가할 수 있음을 고려하고, 재시도에서 확인된 사용량도 합산한다.
- Langfuse 기록 실패가 사용자 답변 생성이나 결과 저장을 실패시키지 않도록 한다.

Timeout 값, 최대 호출 수, 최대 재시도 횟수와 간격은 외부 API의 특성을 확인한 뒤 공식 실행 전에 확정하고 기록한다(세 방식에 동일하게 적용).

### 13.5 테스트 자동화 및 결과 관리 [필수]

테스트는 가능한 한 자동화하여 반복 실행할 수 있도록 한다.

- 단위 테스트: 개별 모듈의 기능 검증
- 통합 테스트: FastAPI, Qdrant, GPT API 및 Langfuse 연결 검증
- 회귀 테스트: 코드 변경 후 기존 기능이 정상 작동하는지 검증
- 장애 테스트: 외부 서비스 실패 및 Timeout 상황 검증 (모의 객체 사용)

테스트 결과에는 최소한 다음 정보를 기록한다.

- 테스트 ID 및 테스트 항목
- 실행 일시
- 실행 환경 및 관련 설정 버전
- 입력 및 실행 조건
- 기대 결과와 실제 결과
- 성공 또는 실패 여부와 실패 원인

외부 API 호출이 필요한 테스트와 모의 객체(Mock)를 이용하는 테스트를 구분하여 불필요한 API 비용을 줄인다.

테스트 상세 항목과 통과 조건은 `08_DEVELOPMENT_PLAN.md`에서 정의한다.

### 13.6 평가 데이터 유입 방지 테스트 [필수]

정답, 근거, `expected_action`, 공격 라벨이 검색·색인·답변 생성 입력에 들어가지 않는지 테스트로 확인한다. (예: 색인 대상이 `documents/*.pdf`뿐인지, 생성 입력에 평가 JSON 필드가 없는지)

### 13.7 서버 운영 안정성 및 부하 테스트 [확장]

다음은 실제 서비스 단계에서 진행한다.

- 서버 시작·종료·재시작 후 Qdrant 및 Langfuse 연결 복구 확인
- Liveness / Readiness 분리
- 반복·중복 요청 처리, 중복 실행 방지를 위한 Idempotency Key (Request ID와 목적을 구분)
- 서로 다른 질문의 동시 요청, 동시 요청 수에 따른 응답시간과 오류율
- 외부 API 호출 제한, 메모리·CPU 사용량

필수 구현 중에도 동일한 질문을 반복해서 요청하는 것은 허용한다(평가 실행에서 필요).

---
## 14. 단계별 구현 우선순위

실행 순서는 `08_DEVELOPMENT_PLAN.md`를 따른다. 먼저 Naive RAG를 질문부터 답변까지 끝까지 연결한 뒤 Hybrid와 Reranking을 같은 API에서 추가한다. STEP 번호는 08번 문서의 번호이다.

| 순서 | STEP | 구현 내용 | 구분 |
|---|---|---|---|
| 1 | 0 | 환경 구성, Git, Langfuse 연결 | [필수] |
| 2 | 1 | PDF 로딩·추출, 페이지 보존, 처리 실패 기록 | [필수] |
| 3 | 2 | 청킹 및 메타데이터 | [필수] |
| 4 | 3 | 임베딩, Qdrant 색인 | [필수] |
| 5 | 4 | Dense Retrieval | [필수] |
| 6 | 7 | FastAPI 기본 구조 및 `/rag/query` | [필수] |
| 7 | 8 | 답변 생성 및 출처 연결 | [필수] |
| 8 | 9 | 되묻기·보류 (Naive RAG 끝까지 동작) | [필수] |
| 9 | 5 | BM25 및 Hybrid Retrieval | [필수] |
| 10 | 6 | Reranker 및 Hybrid + Reranking | [필수] |
| 11 | 10 | Langfuse 통합 검증 | [필수] |
| 12 | 11 | 평가 실행 및 결과 저장 | [필수] |
| 13 | 12 | 공식 RAG 비교 | [필수] |
| 14 | 13 | 보안 비교 | [필수] |
| 15 | 14 | 운영 지표 정리 | [필수] |
| 16 | 15 | README, 최종 결과 정리 | [필수] |
| 17 | 16 | Intermediate / Advanced 데이터셋 확장 | [확장] |
| 18 | 17 | 추가 청킹·검색·프롬프트·모델 실험 | [확장] |
| 19 | - | Redis 캐시 및 운영 성능 실험 | [확장] |
| 20 | 18 | 사용자 Q&A UI 및 실험 결과 공유 화면 | [확장] |

단계별 세부 작업과 완료 조건은 `08_DEVELOPMENT_PLAN.md`에서 정의한다.

---
## 15. 미확정 기술 및 설계 항목

초기 선택은 `06_MODEL_SELECTION.md`에서 확정했다. 아래 표는 현재 상태와 동결 시점이다.

| 항목 | 현재 상태 | 동결 시점 |
|---|---|---|
| Vector DB | Qdrant 확정 (로컬 모드로 시작, 설정으로 서버 모드 전환) | 확정 |
| Embedding Model | multilingual-e5-large 또는 BGE-M3 | STEP 3 소규모 비교 후 동결 |
| Sparse Retrieval | BM25 (rank_bm25 + kiwipiepy), RRF k=60 | 초기값, STEP 5에서 확인 |
| Reranker | BAAI/bge-reranker-v2-m3 | 초기값, STEP 6에서 확인 |
| PDF 추출 라이브러리 | 미정 | STEP 1 (실제 PDF로 확인) |
| Chunking 방식 | 페이지 경계를 넘지 않는 청크, 크기·overlap 미정 | STEP 2, 튜닝은 development에서만 |
| timeout, 최대 호출 수, 재시도 횟수 | 값 미정 | STEP 7~8에서 제안, 공식 실행 전 확정 |
| 결과 저장 방식 | 파일 기반 JSON/JSONL 확정 (`09_DATA_SCHEMA.md`) | 확정 |
| Redis 도입 | 선택적 확장 | 운영 성능 실험 단계 |
| UI 프레임워크 | Streamlit 또는 Gradio ([확장]) | UI 구현 단계 |
| 평가 모델 | gpt-6-luna (초기값) | 공식 실행 전 동결 |

미확정 항목은 구현 편의만을 이유로 임의 확정하지 않는다.

기술 선정 시에는 다음 기준을 고려한다.

- Basic 필수 요구사항 충족 여부
- Intermediate 및 Advanced 확장 가능성
- 구현 복잡도
- 문서 및 유지보수 편의성
- LangChain 및 Langfuse 연동 가능성
- 성능과 비용
- 실험 결과의 재현 가능성
## 16. 완료 기준

본 문서는 다음 조건을 충족하면 아키텍처 설계 초안이 완성된 것으로 본다.

- [ ] 통합 프로젝트 구조가 정의되어 있다.
- [ ] 데이터셋별 선택 및 확장 방식이 정의되어 있다.
- [ ] 세 가지 RAG 구성의 실행 흐름이 정의되어 있다.
- [ ] PDF 색인과 사용자 질문 처리 흐름이 구분되어 있다.
- [ ] 주요 모듈의 책임이 구분되어 있다.
- [ ] API의 기본 역할과 확장 방향이 정의되어 있다.
- [ ] Langfuse 관측 구조의 기본 방향이 정의되어 있다.
- [ ] 평가와 보안 비교의 데이터 흐름이 정의되어 있다.
- [ ] Basic 구현을 보존하는 확장 원칙이 정의되어 있다.
- [ ] 미확정 기술이 임의로 확정되지 않았다.
- [ ] DATA_SCHEMA 및 후속 설계 문서와 역할이 구분되어 있다.

---

## 17. 관련 설계 문서

| 문서 | 역할 |
|---|---|
| `01_PRD.md` | 서비스 목적 및 요구사항 |
| `02_EVALUATION_PLAN.md` | 평가 질문, 지표 및 평가 기준 |
| `03_EXPERIMENT_PLAN.md` | 필수 비교 및 개인 확장 실험 |
| `04_ARCHITECTURE.md` | 시스템 구성, 모듈 책임 및 데이터 흐름 |
| `05_OBSERVABILITY_PLAN.md` | Langfuse Trace 및 운영 관측 설계 |
| `06_MODEL_SELECTION.md` | 생성 모델, 임베딩 및 리랭커 선정 |
| `07_VERSIONING.md` | 데이터, 코드, 설정 및 실험 버전 관리 |
| `08_DEVELOPMENT_PLAN.md` | 단계별 구현 및 테스트 계획 |
| `09_DATA_SCHEMA.md` | 데이터의 구체적인 필드, 자료형 및 API 계약 |
| `10_ADDITIONAL_METRICS.md` | 확장 지표 및 서비스 준비도 게이트 |
| `11_MODEL_SELECTION_GUIDE.md` | 확장 후보 모델·기술 가이드 |
| `12_ACCEPTANCE_CRITERIA.md` | 최종 선택 규칙 |
| `DECISIONS.md` | 승인된 결정과 이유 |

**문서 간 일관성 원칙:** 각 문서는 담당 영역의 세부 사항을 정의하되, 다른 문서의 책임 영역을 임의로 변경하지 않는다.
