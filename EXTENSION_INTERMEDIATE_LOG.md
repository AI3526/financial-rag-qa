# Intermediate RAG 확장 실험 진행 기록

> 브랜치 `extension/intermediate`. Basic 필수 결과(`main`, `results/official-001*`)는 바꾸지 않는다. 결정은 `DECISIONS.md` D-047, 실험 후보는 `project_docs/01_core/03_EXPERIMENT_PLAN.md` §5~10을 따른다.
> 이 파일은 실험이 진행될 때마다 갱신한다. 숫자는 실제로 확인한 값만 적는다.

## 1. 목적과 원칙
- Basic에서 만든 공통 실행 시스템을 재사용해 Intermediate(문서 50개, 약 3,750쪽, 질문 development 30 + additional 30)에서 RAG 방식을 비교하고 개선 후보를 시험한다.
- **기준선**: 먼저 Basic과 같은 설정(추출 pypdf 기본, 청킹 chunk-v1, 임베딩 e5-large, 검색 k=5, 문맥 4,000토큰, 같은 생성·평가 설정)으로 3방식(Dense, Hybrid, Hybrid+Rerank)을 돌려 기준을 만든다. 그 다음 개선 후보를 하나씩 바꿔 비교한다(한 번에 하나만).
- Basic의 hybrid_rerank는 **잠정 선택**(사람 확인 전)이고 Intermediate에서도 최선이라고 가정하지 않는다.
- Langfuse와 Qdrant는 쓰지 않는다(numpy 정확 검색, 로컬 `results/`). 기록은 새 폴더(`results/ext-int-*`, `data/processed_intermediate/`, `data/numpy_index/`)에만 한다.
- 질문 24개는 Basic과 문구가 같지만 정답 근거 문서가 다를 수 있어 질문 ID로만 구분하고 Basic 결과를 재사용하지 않는다.
- 사람 확인은 하지 않고 AI 분류 또는 자동 평가만 쓴다(필요하면 나중에). 이 한계를 모든 결과에 적는다.

## 2. 현재 상태 (2026-11-11)

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 브랜치 생성, 기술 예외 기록(D-047) | 완료 |
| 1 | numpy 정확 검색 구현과 Basic 대조 검증 | 완료 (60문항 상위 5개 집합·순서 모두 일치) |
| 2 | Intermediate PDF 추출 점검(표가 읽히는지) | 예정 |
| 3 | Intermediate 청킹·색인(기준선과 같은 설정) | 예정 |
| 4 | 기준선 3방식 비교(Dense, Hybrid, Hybrid+Rerank) | 예정 |
| 5 | 개선 후보 선정과 실험 계획 | 예정(사용자 확인 후) |

## 3. 기록

### 2026-11-11 — 시작과 numpy 검색 검증
- 확장 브랜치를 만들고 Langfuse 미사용·Qdrant 대신 numpy 정확 검색으로 진행하기로 했다(사용자 승인, D-047).
- 검색 시간 측정(임의 벡터, 1,024차원): 청크 1,056개 0.12ms, 11,000개 0.40ms, 53,000개 3.7ms, 200,000개 12.8ms. 이 규모에서는 numpy로 충분해 FAISS는 쓰지 않는다.
- 구현: `src/index/numpy_store.py`(저장·정확 검색·안정 정렬), `src/retrieval/dense_np.py`(기존 `DenseRetriever`와 같은 출력), `scripts/build_numpy_index.py`, `scripts/verify_numpy_dense.py`, 단위 테스트 5개. `main`의 기존 코드는 수정하지 않았다.
- Basic 색인 재생성: 청크 1,056개, 행렬 4.3MB, 임베딩 22초(GPU). 모델 revision은 기존 동결 값과 같다(`3d7cfbda…`).
- 검증 결과(`results/ext-verify-numpy-001/`): Basic 60문항에서 Qdrant Cloud Dense와 numpy Dense의 상위 5개 **집합 60/60, 순서 60/60 일치, 최대 점수 차이 0.0**. 질문 벡터를 공유한 비교이므로 문서 벡터와 검색 알고리즘의 차이만 반영한다.
- 한계: 이 검증은 Basic 규모(1,056청크)에서만 했다. Intermediate 색인(약 1만 청크)에서의 동등성은 Qdrant 없이 확인할 수 없어, 대신 Basic 검증과 정확 검색이라는 성질에 근거한다.
- 다음: Intermediate PDF 추출 점검. 추출·청킹 스크립트는 Basic 경로가 고정돼 있어 `--dataset` 옵션을 추가해야 한다(기본값은 기존 동작 그대로).
