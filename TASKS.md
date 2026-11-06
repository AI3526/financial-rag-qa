# TASKS.md

작업 목록과 진행 상태. 실행 순서는 `project_docs/01_core/08_DEVELOPMENT_PLAN.md`의 "실행 순서"를 따른다.

상태: TODO / IN_PROGRESS / BLOCKED / REVIEW / DONE
구분: [필수] / [확장]

각 STEP은 시작 전에 작업 범위와 완료 조건을 보고하고, 완료 후 변경 파일·테스트 결과·미해결 사항을 보고한 뒤 사용자 확인을 받고 다음으로 넘어간다.

---

## Phase A. 문서 정합화

| ID | 작업 | 구분 | 상태 |
|---|---|---|---|
| A-00 | 과제 원본 5개 검토 | 필수 | DONE |
| A-01 | 01_PRD 수정 | 필수 | DONE |
| A-02 | 02_EVALUATION_PLAN 수정 | 필수 | DONE |
| A-03 | 03_EXPERIMENT_PLAN 수정 | 필수 | DONE |
| A-04 | 04_ARCHITECTURE 수정 | 필수 | DONE |
| A-05 | 05_OBSERVABILITY_PLAN 수정 | 필수 | DONE |
| A-06 | 06_MODEL_SELECTION 수정 | 필수 | DONE |
| A-07 | 07_VERSIONING 수정 | 필수 | DONE |
| A-08 | 08_DEVELOPMENT_PLAN 수정 | 필수 | DONE |
| A-09 | 09_DATA_SCHEMA 재작성 | 필수 | DONE |
| A-10 | 10_ADDITIONAL_METRICS 수정 | 확장 | DONE |
| A-11 | 11_MODEL_SELECTION_GUIDE 수정 | 확장 | DONE |
| A-12 | 12_ACCEPTANCE_CRITERIA 재작성 | 필수 | DONE |
| A-13 | CLAUDE.md 수정 | 필수 | DONE |
| A-14 | DECISIONS.md, TASKS.md 작성 | 필수 | DONE |
| A-15 | 전체 문서 일관성 검증 (용어, 참조 파일명, 상태 4종, rag_method 값) | 필수 | DONE |

## Phase B/C. 구현 (실행 순서)

| 순서 | STEP | 작업 | 구분 | 상태 | 완료 조건 요약 |
|---|---|---|---|---|---|
| 1 | 0 | 환경 구성: `git init`, uv venv(Python 3.12), `.gitignore`, `.env.example`, `gpt-6-luna`·Langfuse Cloud 연결, 테스트 Trace | 필수 | REVIEW | 프로젝트 실행, API·Langfuse 연결 확인, 비밀정보 미노출 |
| 2 | 1 | PDF 확인·추출: 15개 전체, 페이지 보존, sha256, 처리 실패 기록, 대표 PDF 원문 대조, 개인정보 유무 기록 | 필수 | TODO | 전체 처리, 실패 확인 가능, 원문 대조 |
| 3 | 2 | 청킹·메타데이터: 페이지 경계 비넘김, 최소 메타데이터 | 필수 | TODO | 청크가 `book_id`·`page`에 연결, 대표 청크 대조 |
| 4 | 3 | 임베딩·Qdrant 로컬 색인, e5-large vs BGE-M3 1회 비교, 라이선스·버전 기록 | 필수 | TODO | 색인·조회·원본 페이지 역추적, 승자 동결 |
| 5 | 4 | Dense 검색 | 필수 | TODO | 질문→관련 청크, 빈 결과도 정상 처리 |
| 6 | 7 | FastAPI: `/health`, `/rag/query`, Pydantic, 오류 처리, 필수 안정성 테스트 | 필수 | TODO | 정상·빈 질문·모의 오류 테스트 실행 |
| 7 | 8 | 생성·출처: S-ID 부여, 출력 검증, 출처 변환 | 필수 | TODO | 답변과 출처 연결, 없는 ID 처리 |
| 8 | 9 | 되묻기·보류: 프롬프트·출력 스키마 조정, 예외 질문 테스트 | 필수 | TODO | Naive RAG가 질문→답변까지 동작 |
| 9 | 5 | BM25·Hybrid (RRF) | 필수 | TODO | 세 방식 중 두 번째 연결 |
| 10 | 6 | Reranker·Hybrid+Rerank | 필수 | TODO | 순위 전후 확인 |
| 11 | 10 | Langfuse 통합 검증 | 필수 | TODO | 대표 요청 1건 추적 |
| 12 | 11 | 평가: Hit/Recall, RAGAS Faithfulness, Judge 4항목 | 필수 | TODO | 결과 저장, 대표 사례 원문 대조 |
| 13 | 12 | 공식 비교: 설정 동결 후 60문항 × 3방식 | 필수 | TODO | 비교표, 최종 선택 근거 |
| 14 | 13 | 보안 비교: FPAIR01·FPAIR02 × 기본/방어 | 필수 | TODO | 답변 4개와 판정 |
| 15 | 14 | 운영 지표 정리 | 필수 | TODO | 응답시간·오류율·토큰 |
| 16 | 15 | README, 최종 결과, 발표자료 | 필수 | TODO | 재실행 가능한 문서화 |

## 확장 (필수 완료 후)

| STEP | 작업 | 구분 | 상태 |
|---|---|---|---|
| 16 | Intermediate / Advanced 확장 (Qdrant 서버 모드 전환 포함) | 확장 | TODO |
| 17 | 추가 모델·검색·프롬프트 실험, 자체 clarify·abstain 질문 추가 | 확장 | TODO |
| 18 | UI, 실험 결과 대시보드, 캐시, 서비스 준비도 게이트 적용 | 확장 | TODO |

## 열려 있는 항목 (해당 STEP에서 확정)

- timeout / 최대 호출 수 / 재시도 횟수 값 (STEP 7~8에서 제안)
- `gpt-6-luna` 지원 파라미터, 임베딩 승자 (STEP 0, 3)
- 표가 페이지 경계에서 잘리는 문제 (STEP 1~2)
- Langfuse SDK 버전, `request_id`→`trace_id` 생성 방식 (STEP 0)
- 모델·패키지 정확 버전과 라이선스 (STEP 3, `DECISIONS.md`에 기록)
