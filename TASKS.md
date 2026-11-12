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
| 1 | 0 | 환경 구성: `git init`, uv venv(Python 3.12), `.gitignore`, `.env.example`, `gpt-6-luna`·Langfuse Cloud 연결, 테스트 Trace | 필수 | DONE | 프로젝트 실행, API·Langfuse 연결 확인, 비밀정보 미노출 |
| 2 | 1 | PDF 확인·추출: 15개 전체, 페이지 보존, sha256, 처리 실패 기록, 대표 PDF 원문 대조, 개인정보 유무 기록 | 필수 | DONE | 전체 처리, 실패 확인 가능, 원문 대조, Langfuse 기록(Trace 1 + Span 15), 이메일 8건은 공개 가상 연락처로 개인정보 해당 없음(D-022) |
| 3 | 2 | 청킹·메타데이터: 페이지 경계 비넘김, 최소 메타데이터 (D-025) | 필수 | REVIEW | 청크가 `book_id`·`page`에 연결, 대표 청크 대조 |
| 4 | 3 | 임베딩·Qdrant(Cloud 우선, D-010) 색인, e5-large vs BGE-M3 1회 비교, 라이선스·버전 기록 (D-026, D-027) | 필수 | REVIEW | 색인·조회·원본 페이지 역추적 확인, 승자 e5 동결(Recall@5 0.7649 vs 0.7619, Hit@5는 BGE-M3 0.9286) |
| 5 | 4 | Dense 검색 (D-028). 청크 크기·overlap 튜닝은 [확장]이라 제외(03 §6) | 필수 | REVIEW | 질문→관련 청크, 원본 PDF 페이지 연결, 점수·순위 확인, 빈 질문·빈 결과·저장소 오류 처리 |
| 6 | 7 | FastAPI: `/health`, `/rag/query`, Pydantic, 오류 처리, 필수 안정성 테스트 (D-031) | 필수 | REVIEW | 정상·빈 질문·모의 오류 테스트 실행 완료, `traceparent` HTTP 구간으로 실험 Trace에 이어 붙음 확인 |
| 7 | 8 | 생성·출처: S-ID 부여, 출력 검증, 출처 변환 (D-032) | 필수 | REVIEW | 실제 gpt-6-luna로 6개 질문 답변과 출처 연결 확인(AI가 2건 원문 비교, **사람 확인은 미완료**). Langfuse Generation이 `rag-query` 아래에 붙음 확인(캐시 토큰 발견 기록). 단가를 `configs/pricing.json`에 기록하고 요청별 비용 계산(D-033). 남은 것: 생성 설정은 STEP 9 확인 후 동결, Langfuse 모델 단가 등록 |
| 8 | 9 | 되묻기·보류: 프롬프트·출력 스키마 조정, 예외 질문 테스트 (D-034) | 필수 | REVIEW | development 30문항 상태 26/30 일치(불일치 4건은 검색 한계), 예외 질문 6/6, 프롬프트 answer-v2(사용자가 문구 확인·승인). 조정 횟수 제한 없음(문구 변경은 사전 승인). 프롬프트 동결은 공식 실행 전 |
| 9 | 5 | BM25·Hybrid (RRF) (D-035) | 필수 | REVIEW | BM25(토크나이저 whitespace 동결)·RRF·Hybrid 구현, `rag_method=hybrid` 연결, Dense·BM25·Hybrid 비교 완료(D-036: Hybrid는 Hit@5 0.9286으로 최고지만 Recall@5는 Dense보다 낮음). 실제 Langfuse에서 `hybrid-retrieve` 아래 Dense·BM25 Span 연결 확인 |
| 10 | 6 | Reranker·Hybrid+Rerank (D-037) | 필수 | REVIEW | bge-reranker-v2-m3(apache-2.0, 확인한 revision 기록) 연결, 재정렬 전후 순위·점수 기록, 4방식 비교 완료(Hybrid+Rerank가 Recall@5 0.8095로 최고, Hit@5는 Hybrid보다 낮음, 질문당 약 0.5초 추가). 실제 Langfuse에서 `rerank-retrieve` 아래 Span 연결 확인 |
| 11 | 10 | Langfuse 통합 검증 (D-038) | 필수 | REVIEW | 대표 요청 6건이 24개 검사 통과(Trace 연결, 식별자, 최종 문맥 구분, 모델 입출력·사용량, 오류 유형·단계·재시도). config_version 정의·공개 키(SDK 기본 동작) 사용자 승인. 평가 LLM 호출과 서비스 기록의 구분은 Faithfulness로 확인함(D-039), Judge 호출은 Judge 구현 후 확인 |
| 12 | 11 | 평가: Hit/Recall, RAGAS Faithfulness, Judge 4항목 (D-039, D-040) | 필수 | REVIEW | Faithfulness 완료(격리 환경, 승인된 1단계 프롬프트, development 30개 채점·판정 이유 저장). 평가 호출의 Langfuse 구분 확인 완료(3건). Judge 4항목 완료(승인된 `judge-v1`, development 30개 로컬 채점, D-040). 세 방식 development 점검 완료(D-041, 공식 결과 아님). Judge 호출의 Langfuse 기록 확인 완료(3건), 검색 평가(Hit@k·Recall@k) 공식 형식 통합 완료(D-042). 사람 대조: Judge 필수 5사례 완료(판정 19/20 동의, D-043), Faithfulness는 사용자가 채팅으로 판단을 알림(사례 4는 (1)(2)번 문장 모두 주장으로 세므로 N/A에 불일치, 사례 6은 0이 아니라 1로 봄, 7과 나머지는 문제 없음; 사람도 4·6·7은 판단이 어려웠음), 평가 기준 확정. `judge-v2` 승인·구현·development 재채점 완료(D-044, 사람 기준 20/20, 근거 충족률 변화, 진단 추가), Faithfulness v1 유지 확정(한계 기록). 남은 것: **사용자 확인, 커밋, STEP 12 시작 전 Judge 채점 회차 수 결정과 설정 동결** |
| 13 | 12 | 공식 비교: 설정 동결 후 60문항 × 3방식 | 필수 | REVIEW | 공식 실행·평가·비교표 완료, Basic 방식 hybrid_rerank **잠정 선택**(D-046, 사람 확인 후 확정). 사람 확인(중대 오류 표본, 대표 사례)은 사용자 결정으로 보류해 AI 분류만 사용(요건 미충족 예외 기록). STEP 13~15는 사람 확인 후로 보류. 남은 것: 커밋, Langfuse Experiments 비교 화면 원인 조사, `post_scores.py --apply` |
| 14 | 13 | 보안 비교: FPAIR01·FPAIR02 × 기본/방어 | 필수 | TODO | 답변 4개와 판정 |
| 15 | 14 | 운영 지표 정리 | 필수 | REVIEW | `official-001` 일반 평가 180건으로 정리 완료(`results/official-001-ops/operational.md`·`operational.json`·`per_question_ops.csv`): 세 방식 시스템 오류 0, 평균 응답시간 dense 2,626 / hybrid 2,540 / hybrid_rerank 2,974ms, 입력·출력 토큰 합계와 비용(약 0.0175~0.0177 USD)을 확인한 요청 60/60, 평가 호출 사용량 분리(D-048). 사람 확인 후 선택이 확정되면 '잠정' 표시를 바꿈 |
| 16 | 15 | README, 최종 결과, 발표자료 | 필수 | TODO | 재실행 가능한 문서화 |

## 확장 (필수 완료 후)

| STEP | 작업 | 구분 | 상태 |
|---|---|---|---|
| 16 | Intermediate / Advanced 확장 (Qdrant 서버 모드 전환 포함) | 확장 | TODO |
| 17 | 추가 모델·검색·프롬프트 실험, 자체 clarify·abstain 질문 추가 | 확장 | TODO |
| 18 | UI, 실험 결과 대시보드, 캐시, 서비스 준비도 게이트 적용 | 확장 | TODO |

## 열려 있는 항목 (해당 STEP에서 확정)

- timeout / 최대 호출 수 / 재시도 횟수 값: 초기값 제안됨(D-030, 지연 측정 근거), STEP 8 실제 프롬프트로 재측정 후 공식 실행 전 확정
- 확인 완료: `gpt-6-luna` 지원 파라미터(STEP 0), 임베딩 승자 e5(STEP 3, D-027)
- Qdrant Cloud 무료 플랜의 용량·비활성 정지 정책 (미확인, 확인 후 기록)
- 서비스 응답시간(STEP 14)에는 Cloud 네트워크 시간이 포함됨을 보고서에 명시
- 표가 페이지 경계에서 잘리는 문제 (STEP 1~2)
- 확인 완료: Langfuse SDK 5.2.0(STEP 0), 요청·실험 Trace 연결과 환경 규칙(D-029, spike). HTTP 구간 전달은 STEP 7에서 확인함(D-031). 요청당 관측 수 실측은 STEP 10에서 함(D-038, 공식 실행 약 3,000유닛 추정). 점수 방식(`create_score` 후붙임)과 비교 화면 표시는 점수가 생기는 STEP 11에서 확인
- 사용자가 Langfuse 화면에서 임시 데이터셋 `spike-trace-link` 삭제 (SDK에 삭제 API 없음)
- 모델·패키지 정확 버전과 라이선스 (STEP 3, `DECISIONS.md`에 기록)
