from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict


class RefactorService:
    MAX_CONTEXT_CHARS = 16000

    def __init__(self, *, workspace_root: Path, run_llm_call):
        self.workspace_root = workspace_root.resolve()
        self.run_llm_call = run_llm_call

    async def run_refactor(self, *, user_text: str, match: Dict[str, Any]) -> Dict[str, Any] | None:
        file_path = (match or {}).get("file_path")
        if not file_path:
            return None
        source_file = Path(file_path).resolve()
        if not str(source_file).startswith(str(self.workspace_root)):
            return None
        rel_in_repo = self._resolve_relative_path(source_file)
        if not rel_in_repo:
            return None
        if not source_file.exists():
            return None
        original_code = source_file.read_text(encoding="utf-8", errors="ignore")
        trimmed_code = original_code
        if len(trimmed_code) > self.MAX_CONTEXT_CHARS:
            trimmed_code = trimmed_code[: self.MAX_CONTEXT_CHARS]
        snippet = self._prepare_snippet((match or {}).get("content"))
        prompt = self._build_prompt(
            user_text=user_text,
            relative_path=str(rel_in_repo),
            snippet=snippet,
            code=trimmed_code,
        )
        response = await self.run_llm_call(prompt, task="refactor_agent", max_new_tokens=1200)
        parsed = self._safe_json_object(response)
        if not parsed or not parsed.get("updated_code"):
            return None
        updated_code = parsed["updated_code"].rstrip() + "\n"
        summary = (parsed.get("summary") or "").strip()
        diff_preview = self._build_diff_preview(original_code, updated_code)
        return {
            "summary": summary or "리팩터링 코드가 생성되었습니다.",
            "file_path": str(source_file),
            "relative_path": str(rel_in_repo),
            "original_snippet": snippet,
            "updated_code": updated_code,
            "diff_preview": diff_preview,
        }

    def _resolve_relative_path(self, source_file: Path) -> Path | None:
        try:
            rel = source_file.relative_to(self.workspace_root)
        except ValueError:
            return None
        if not rel.parts:
            return None
        return rel

    @staticmethod
    def _prepare_snippet(snippet: str | None) -> str:
        text = (snippet or "").strip()
        if not text:
            return ""
        if len(text) > 1200:
            return text[:1200] + "\n... (truncated)"
        return text

    def _build_prompt(self, *, user_text: str, relative_path: str, snippet: str, code: str) -> str:
        snippet_block = snippet.strip() or "(snippet not provided)"
        return (
            "다음은 사용자의 리팩터링 요청과 대상 파일 정보입니다.\n\n"
            f"- 파일 경로: {relative_path}\n"
            f"- 사용자 요청: {user_text}\n\n"
            "[관련 코드 스니펫]\n"
            f"{snippet_block}\n\n"
            "[원본 전체 코드]\n"
            "```python\n"
            f"{code}\n"
            "```\n\n"
            "요청에 맞게 전체 파일을 리팩터링하거나 필요한 수정을 모두 반영하세요. "
            "결과는 JSON 한 줄로만 답하며, 형식은 "
            '{"updated_code": "<수정된 전체 코드>", "summary": "<변경사항 요약 (한국어 한 문장)>"} 입니다.'
        )

    @staticmethod
    def _safe_json_object(text: str) -> Dict[str, Any] | None:
        if not text:
            return None
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            payload = text[start:end]
            return json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _build_diff_preview(original: str, updated: str) -> str:
        diff = difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile="original",
            tofile="refac",
            n=3,
        )
        preview = "\n".join(diff)
        if len(preview) > 2000:
            preview = preview[:2000] + "\n... (diff truncated)"
        return preview or "(diff not available)"


__all__ = ["RefactorService"]
