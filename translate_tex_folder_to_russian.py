#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "gpt-5"
DEFAULT_TRANSLATE_EXTENSIONS = (".tex",)

# Environments whose content should be copied verbatim.
PROTECTED_ENVS = [
    "equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "array", "matrix", "pmatrix",
    "bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "cases", "split", "math",
    "displaymath", "tikzpicture", "lstlisting", "verbatim", "Verbatim", "minted",
    "pythoncode", "figure", "figure*", "table", "table*"
]

# Commands that should generally remain untouched.
COPY_COMMANDS = {
    "cite", "citet", "citep", "Cite", "Citet", "Citep", "nocite",
    "ref", "eqref", "autoref", "cref", "Cref", "pageref", "label",
    "url", "href", "doi", "includegraphics", "input", "include", "bibliography",
    "bibliographystyle", "footnote", "item", "begin", "end"
}

TRANSLATION_INSTRUCTIONS = r"""
You are translating LaTeX source code from English to Russian.
Return valid LaTeX.

Hard rules:
1. Preserve all LaTeX commands, environments, labels, references, citation keys, bibliography keys, file paths, macro names, and math exactly.
2. Translate only human-readable prose: normal sentences, section titles, captions, abstract text, theorem/proof text, list item text.
3. Do not change text inside math mode or protected blocks.
4. Do not rewrite formulas or alter spacing in ways that break LaTeX.
5. Keep comments, indentation, and blank-line paragraph structure whenever practical.
6. Output only the translated LaTeX fragment, with no explanations.
7. Russian should sound natural and academic, not word-for-word.
8. Preserve names of people, organizations, software, datasets, citation keys, and email addresses unless there is an obvious standard Russian form.
9. If placeholder tokens like @@KEEP_123@@ appear, preserve them exactly in place. Do not delete, renumber, merge, or move them.
""".strip()


class TranslationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate .tex files in a folder to Russian while preserving LaTeX.")
    p.add_argument("source", help="Source folder or single .tex file")
    p.add_argument("-o", "--output", default=None, help="Output folder. Default: <source>_ru")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    p.add_argument("--max-chars", type=int, default=6000, help="Approximate chunk size sent to the model")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between requests, seconds")
    p.add_argument(
        "--translate-ext",
        default=",".join(DEFAULT_TRANSLATE_EXTENSIONS),
        help="Comma-separated file extensions to translate when source is a folder. Default: .tex",
    )
    p.add_argument(
        "--copy-other-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-translated files to the output folder so the translated archive stays self-contained.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing translated files")
    return p.parse_args()


def ensure_openai_available() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Python package 'openai' is not installed. Install it first, for example: "
            "python3 -m pip install openai"
        ) from exc


def ensure_api_key() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    if key.startswith("sess-"):
        raise SystemExit("OPENAI_API_KEY looks like a session token (sess-...), not a Platform API key.")


def make_client() -> Any:
    ensure_openai_available()
    ensure_api_key()
    from openai import OpenAI

    return OpenAI()


TOKEN_RE = re.compile(r"@@KEEP_(\d+)@@")


def protect_patterns(text: str) -> Tuple[str, List[str]]:
    kept: List[str] = []

    def keep(match: re.Match[str]) -> str:
        kept.append(match.group(0))
        return f"@@KEEP_{len(kept)-1}@@"

    # Protected environments
    env_names = "|".join(re.escape(x) for x in sorted(PROTECTED_ENVS, key=len, reverse=True))
    if env_names:
        env_pat = re.compile(rf"\\begin\{{({env_names})\}}.*?\\end\{{\1\}}", re.DOTALL)
        text = env_pat.sub(keep, text)

    # Display math and inline math
    patterns = [
        re.compile(r"\\\[.*?\\\]", re.DOTALL),
        re.compile(r"\\\(.*?\\\)", re.DOTALL),
        re.compile(r"\$\$.*?\$\$", re.DOTALL),
        re.compile(r"(?<!\\)\$(?:\\.|[^$\\])+?(?<!\\)\$", re.DOTALL),
    ]
    for pat in patterns:
        text = pat.sub(keep, text)

    # Comments
    text = re.sub(r"(?m)^([^\n%]*?)(?<!\\)(%.*)$", lambda m: m.group(1) + keep(re.match(r".*", m.group(2))), text)

    return text, kept


def restore_patterns(text: str, kept: List[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return kept[int(match.group(1))]
    return TOKEN_RE.sub(repl, text)


def ensure_no_placeholders(text: str) -> None:
    leftovers = TOKEN_RE.findall(text)
    if leftovers:
        sample = ", ".join(f"@@KEEP_{i}@@" for i in leftovers[:5])
        raise TranslationError(
            f"Unrestored placeholder tokens remain in translated output: {sample}"
        )


SECTION_CMD_RE = re.compile(
    r"(\\(?:part|chapter|section|subsection|subsubsection|paragraph|subparagraph|title|author|caption|chapter\*|section\*|subsection\*|subsubsection\*)\s*(?:\[[^\]]*\])?\{)",
    re.DOTALL,
)


def split_text_for_translation(text: str, max_chars: int) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    blocks = re.split(r"(\n\s*\n)", text)
    for block in blocks:
        if not block:
            continue
        piece_len = len(block)
        if piece_len > max_chars and block.strip() and block.strip() != "":
            if current:
                parts.append("".join(current))
                current, current_len = [], 0
            parts.extend(split_long_block(block, max_chars))
            continue
        if current_len + piece_len > max_chars and current:
            parts.append("".join(current))
            current, current_len = [], 0
        current.append(block)
        current_len += piece_len
    if current:
        parts.append("".join(current))
    return [p for p in parts if p]


def split_long_block(block: str, max_chars: int) -> List[str]:
    # Prefer sentence-like splits, then hard fallback.
    candidates = re.split(r"(?<=[.!?])(?=\s+[A-ZА-Я\\])", block)
    if len(candidates) == 1:
        return [block[i:i + max_chars] for i in range(0, len(block), max_chars)]
    out: List[str] = []
    cur = ""
    for c in candidates:
        if len(cur) + len(c) > max_chars and cur:
            out.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def is_latexy(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith("%"):
        return True
    if s.startswith("\\") and not re.search(r"\{.*[A-Za-zА-Яа-я].*\}", s):
        return True
    return False


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def translate_fragment(client: Any, model: str, fragment: str, retries: int = 5) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=TRANSLATION_INSTRUCTIONS,
                input=fragment,
            )
            output = (response.output_text or "").strip("\n")
            if not output:
                raise TranslationError("Empty model output")
            return output
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = min(2 ** attempt, 20)
            print(f"[warn] Translation request failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            time.sleep(wait)
    raise TranslationError(str(last_error))


def translate_tex_text(client: Any, model: str, text: str, max_chars: int, sleep_s: float) -> str:
    text = normalize_newlines(text)
    protected_text, kept = protect_patterns(text)
    chunks = split_text_for_translation(protected_text, max_chars)
    out_chunks: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if chunk.strip() and re.search(r"[A-Za-zА-Яа-я]", chunk):
            print(f"    [info] chunk {i}/{len(chunks)}")
            out = translate_fragment(client, model, chunk)
            out_chunks.append(out)
            if sleep_s > 0:
                time.sleep(sleep_s)
        else:
            out_chunks.append(chunk)
    joined = "".join(out_chunks)
    restored = restore_patterns(joined, kept)
    ensure_no_placeholders(restored)
    return restored


def parse_extensions(raw: str) -> Tuple[str, ...]:
    exts: List[str] = []
    for item in raw.split(","):
        ext = item.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext.lower())
    if not exts:
        raise SystemExit("At least one extension must be provided via --translate-ext.")
    return tuple(dict.fromkeys(exts))


def iter_translatable_files(source: Path, extensions: Sequence[str]) -> List[Path]:
    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise SystemExit(
                f"If source is a file, its extension must be one of: {', '.join(extensions)}"
            )
        return [source]
    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def build_output_root(source: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    if source.is_file():
        return source.with_suffix("").with_name(source.stem + "_ru")
    return source.parent / f"{source.name}_ru"


def write_output(src: Path, source_root: Path, out_root: Path, content: str) -> Path:
    if source_root.is_file():
        out_path = out_root.with_suffix(src.suffix) if out_root.suffix == "" else out_root
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        rel = src.relative_to(source_root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def copy_other_files(
    source_root: Path,
    out_root: Path,
    translated_files: Sequence[Path],
    force: bool,
) -> int:
    if source_root.is_file():
        return 0

    translated_set = {p.resolve() for p in translated_files}
    copied = 0
    for src in sorted(p for p in source_root.rglob("*") if p.is_file()):
        if src.resolve() in translated_set:
            continue
        dst = out_root / src.relative_to(source_root)
        if dst.exists() and not force:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    translate_exts = parse_extensions(args.translate_ext)
    files = iter_translatable_files(source, translate_exts)
    if not files:
        raise SystemExit(f"No files found with extensions: {', '.join(translate_exts)}")

    out_root = build_output_root(source, args.output).resolve()
    client = make_client()

    print(f"[info] Found {len(files)} translatable file(s)")
    for idx, src in enumerate(files, start=1):
        print(f"[info] File {idx}/{len(files)}: {src}")
        if source.is_file():
            target_ext = source.suffix if out_root.suffix == "" else out_root.suffix
            target = out_root.with_suffix(target_ext)
        else:
            target = out_root / src.relative_to(source)
        if target.exists() and not args.force:
            print(f"    [skip] exists: {target}")
            continue
        text = src.read_text(encoding="utf-8")
        translated = translate_tex_text(client, args.model, text, args.max_chars, args.sleep)
        out_path = write_output(src, source, out_root, translated)
        print(f"    [ok] saved: {out_path}")

    if args.copy_other_files:
        copied = copy_other_files(source, out_root, files, args.force)
        print(f"[info] Copied {copied} non-translated file(s)")

    print("[done]")


if __name__ == "__main__":
    main()
