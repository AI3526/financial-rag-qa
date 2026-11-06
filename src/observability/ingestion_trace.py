"""문서 처리 실행(추출·청킹 등)의 Langfuse 기록.

- 실행 1회 = Trace 1개 (`document-ingestion`, `document-chunking`), 문서마다 Span 1개.
  질문 처리 Trace(`rag-query`)와 구분한다 (08_DEVELOPMENT_PLAN.md 5절, 05_OBSERVABILITY_PLAN.md).
- Span 이름에 원본 PDF 파일명을 넣고, 출력에 문서 요약과 텍스트를 담는다. 유닛은 Trace/Observation 개수로 세므로
  (페이지·청크마다 Span을 만들지 않아) 한 번에 Trace 1 + Span 15 = 16유닛이다.
- Langfuse가 없거나 실패해도 처리는 멈추지 않는다 (04_ARCHITECTURE.md 13.4).
- 개인정보가 포함된 데이터를 처리하기 전에는 Langfuse `mask` 함수를 먼저 구현해야 한다 (DECISIONS.md D-022).
"""
from __future__ import annotations

import logging
import re
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from src.ingest.extract import extract_pdf

logger = logging.getLogger(__name__)

# Langfuse environment 값 규칙: 소문자·숫자·하이픈·밑줄, 'langfuse'로 시작하지 않음 (SDK 문서)
_ENVIRONMENT_RE = re.compile(r"^(?!langfuse)[a-z0-9_-]+$")


class _RunTracer:
    """실행 단위 Trace를 열고 닫는 공통 기반. 하위 클래스가 TRACE_NAME, TAGS와 문서별 메서드를 정한다."""

    TRACE_NAME = "document-run"
    TAGS: list[str] = []

    def __init__(self, langfuse, meta: dict[str, Any] | None = None, environment: str = "development"):
        """environment: Langfuse의 기본 환경 필드. 프로젝트의 `track` 값(development/official/extension)과 같게 쓴다."""
        if not _ENVIRONMENT_RE.fullmatch(environment):
            raise ValueError(f"올바르지 않은 environment 값: {environment!r}")
        self.langfuse = langfuse
        self.environment = environment
        self.meta = {k: str(v) for k, v in (meta or {}).items()}
        self.run_id = str(uuid.uuid4())
        self.trace_id: str | None = None
        self._stack: ExitStack | None = None
        self.documents_recorded = 0

    # --- 실행 단위 Trace ---
    def __enter__(self):
        if self.langfuse is None:
            return self
        try:
            from langfuse import propagate_attributes

            self.trace_id = self.langfuse.create_trace_id(seed=self.run_id)
            stack = ExitStack()
            stack.enter_context(
                propagate_attributes(
                    trace_name=self.TRACE_NAME,
                    environment=self.environment,
                    tags=list(self.TAGS),
                    metadata={"run_id": self.run_id, "track": self.environment, **self.meta},
                )
            )
            stack.enter_context(
                self.langfuse.start_as_current_observation(
                    trace_context={"trace_id": self.trace_id},
                    name=self.TRACE_NAME,
                    as_type="span",
                    input={"run_id": self.run_id, **self.meta},
                )
            )
            self._stack = stack
        except Exception as exc:  # noqa: BLE001 - 관측 실패가 처리를 막지 않게 한다
            logger.warning("Langfuse Trace 시작 실패: %s", type(exc).__name__)
            self._stack = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._stack is not None:
            try:
                self._stack.close()
            except Exception as err:  # noqa: BLE001
                logger.warning("Langfuse Trace 종료 실패: %s", type(err).__name__)
        if self.langfuse is not None:
            try:
                self.langfuse.flush()
            except Exception as err:  # noqa: BLE001
                logger.warning("Langfuse flush 실패: %s", type(err).__name__)
        return False

    def trace_url(self) -> str | None:
        if self.langfuse is None or self.trace_id is None:
            return None
        try:
            return self.langfuse.get_trace_url(trace_id=self.trace_id)
        except Exception:  # noqa: BLE001
            return None

    # --- 문서 1개 = Span 1개 ---
    def _traced(
        self,
        span_name: str,
        span_input: dict,
        span_metadata: dict,
        compute: Callable[[], Any],
        summarize: Callable[[Any], tuple[dict, bool, str | None]],
    ) -> Any:
        """compute()를 실행하고 결과를 Span으로 기록한다. 반환값은 compute()의 결과.

        summarize(result) -> (출력, 실패 여부, 상태 메시지). Langfuse 오류는 삼키고 처리 결과에는 영향을 주지 않는다.
        """
        span_stack: ExitStack | None = None
        span = None
        if self.langfuse is not None and self._stack is not None:
            try:
                span_stack = ExitStack()
                span = span_stack.enter_context(
                    self.langfuse.start_as_current_observation(
                        name=span_name, as_type="span", input=span_input, metadata=span_metadata,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse Span 시작 실패: %s", type(exc).__name__)
                span_stack, span = None, None

        result = compute()

        if span is not None:
            try:
                output, failed, status_message = summarize(result)
                span.update(
                    output=output,
                    level="ERROR" if failed else "DEFAULT",
                    status_message=status_message if failed else None,
                )
                self.documents_recorded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse Span 기록 실패: %s", type(exc).__name__)
            finally:
                try:
                    span_stack.close()
                except Exception:  # noqa: BLE001
                    pass
        return result


class IngestionTracer(_RunTracer):
    """PDF 추출 실행 기록. `with IngestionTracer(langfuse, meta) as tracer:` 안에서 tracer.extract(...)를 호출한다."""

    TRACE_NAME = "document-ingestion"
    TAGS = ["ingestion"]

    def extract(self, pdf_path: Path, **kwargs) -> tuple[dict, list[dict]]:
        """extract_pdf를 실행하고 결과를 Span으로 기록한다. 반환값은 extract_pdf와 같다."""

        def summarize(result):
            doc, pages = result
            failed = doc["extraction_status"] != "success"
            output = {
                "summary": {k: doc[k] for k in (
                    "book_id", "file_name", "file_sha256", "file_size", "page_count",
                    "extraction_status", "extraction_seconds", "error_message",
                )},
                "pages": [
                    {"page": p["page"], "status": p["extraction_status"], "text": p["text"]} for p in pages
                ],
            }
            return output, failed, doc["error_message"]

        return self._traced(
            # 트리 화면에서 어떤 PDF인지 바로 보이도록 이름에 원본 파일명을 넣는다
            f"extract-pdf {pdf_path.name}",
            {"book_id": pdf_path.stem, "file_name": pdf_path.name},
            {"book_id": pdf_path.stem},
            lambda: extract_pdf(pdf_path, **kwargs),
            summarize,
        )


class ChunkingTracer(_RunTracer):
    """청킹 실행 기록. `with ChunkingTracer(langfuse, meta) as tracer:` 안에서 tracer.chunk(...)를 호출한다."""

    TRACE_NAME = "document-chunking"
    TAGS = ["chunking"]

    def chunk(
        self,
        doc: dict,
        compute: Callable[[], tuple[list[dict], list[dict]]],
        params: dict,
    ) -> tuple[list[dict], list[dict]]:
        """compute()가 만든 (청크, 경고)를 문서 Span으로 기록한다. 반환값은 compute()의 결과."""

        def summarize(result):
            chunks, warnings = result
            sizes = [len(c["chunk_text"]) for c in chunks]
            failed = len(chunks) == 0
            output = {
                "summary": {
                    "book_id": doc["book_id"],
                    "file_name": doc["file_name"],
                    "chunk_count": len(chunks),
                    "pages_chunked": len({c["page"] for c in chunks}),
                    "avg_chars": round(sum(sizes) / len(sizes), 1) if sizes else None,
                    "max_chars": max(sizes) if sizes else None,
                    "warnings": warnings,
                },
                "chunks": [
                    {"chunk_id": c["chunk_id"], "page": c["page"], "chars": len(c["chunk_text"]),
                     "text": c["chunk_text"]}
                    for c in chunks
                ],
            }
            return output, failed, "no_chunk" if failed else None

        return self._traced(
            f"chunk-pdf {doc['file_name']}",
            {"book_id": doc["book_id"], "file_name": doc["file_name"], "params": params},
            {"book_id": doc["book_id"]},
            compute,
            summarize,
        )
