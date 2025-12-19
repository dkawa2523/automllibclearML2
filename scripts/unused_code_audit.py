#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "cleanup"

_VULTURE_LINE_RE = re.compile(r"^(?P<path>[^:]+):(?P<line>\\d+):\\s*(?P<msg>.*)\\s+\\(confidence\\s+(?P<conf>\\d+)%\\)$")


def _run(cmd: List[str], *, env: Dict[str, str], cwd: Path, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=bool(capture),
        check=False,
    )


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    # Ensure coverage runs do not require ClearML server connectivity.
    env.setdefault("CLEARML_OFFLINE_MODE", "1")
    env.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
    # Avoid PipelineController ping/queue checks if someone adds pipeline commands later.
    env.setdefault("AUTO_ML_SKIP_CLEARML_PING", "1")
    env.setdefault("AUTO_ML_SKIP_CLEARML_QUEUE_CHECK", "1")
    return env


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class VultureItem:
    path: str
    line: int
    confidence: int
    message: str


def _parse_vulture(stdout: str) -> List[VultureItem]:
    items: List[VultureItem] = []
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _VULTURE_LINE_RE.match(line)
        if not m:
            continue
        try:
            items.append(
                VultureItem(
                    path=str(m.group("path")),
                    line=int(m.group("line")),
                    confidence=int(m.group("conf")),
                    message=str(m.group("msg")).strip(),
                )
            )
        except Exception:
            continue
    return items


def run_vulture(*, out_dir: Path, package: str, min_confidence: int) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "vulture", package, f"--min-confidence={int(min_confidence)}"]
    proc = _run(cmd, env=_base_env(), cwd=REPO_ROOT, capture=True)
    _write_text(out_dir / "vulture.txt", (proc.stdout or "") + (proc.stderr or ""))
    items = _parse_vulture(proc.stdout or "")
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "items": [item.__dict__ for item in items],
    }


def _coverage_cmd_for_python_args(python_args: List[str], *, append: bool) -> List[str]:
    base = [sys.executable, "-m", "coverage", "run", "--source=automl_lib"]
    if append:
        base.append("-a")
    base.extend(python_args)
    return base


def run_coverage(*, out_dir: Path, coverage_commands: List[List[str]]) -> Dict[str, Any]:
    env = _base_env()

    erase = _run([sys.executable, "-m", "coverage", "erase"], env=env, cwd=REPO_ROOT, capture=True)
    _write_text(out_dir / "coverage_erase.txt", (erase.stdout or "") + (erase.stderr or ""))

    runs: List[Dict[str, Any]] = []
    first = True
    for args in coverage_commands:
        cmd = _coverage_cmd_for_python_args(args, append=not first)
        first = False
        proc = _run(cmd, env=env, cwd=REPO_ROOT, capture=True)
        runs.append(
            {
                "cmd": cmd,
                "returncode": int(proc.returncode),
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
            }
        )

    # Always write human-readable report + machine-readable JSON
    report = _run([sys.executable, "-m", "coverage", "report", "-m"], env=env, cwd=REPO_ROOT, capture=True)
    _write_text(out_dir / "coverage_report.txt", (report.stdout or "") + (report.stderr or ""))

    cov_json_path = out_dir / "coverage.json"
    cov_json = _run([sys.executable, "-m", "coverage", "json", "-o", str(cov_json_path)], env=env, cwd=REPO_ROOT, capture=True)
    _write_text(out_dir / "coverage_json.txt", (cov_json.stdout or "") + (cov_json.stderr or ""))

    data: Optional[Dict[str, Any]] = None
    try:
        if cov_json_path.exists():
            data = json.loads(cov_json_path.read_text(encoding="utf-8"))
    except Exception:
        data = None

    return {
        "runs": runs,
        "coverage_report_returncode": int(report.returncode),
        "coverage_json_returncode": int(cov_json.returncode),
        "coverage_json_path": str(cov_json_path),
        "coverage": data,
    }


def _extract_zero_coverage_files(coverage_json: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(coverage_json, dict):
        return []
    files = coverage_json.get("files")
    if not isinstance(files, dict):
        return []
    out: List[Dict[str, Any]] = []
    for path, meta in files.items():
        if not isinstance(meta, dict):
            continue
        summary = meta.get("summary")
        if not isinstance(summary, dict):
            continue
        try:
            pct = float(summary.get("percent_covered", 0.0))
        except Exception:
            pct = 0.0
        if pct != 0.0:
            continue
        out.append(
            {
                "path": str(path),
                "percent_covered": pct,
                "num_statements": int(summary.get("num_statements", 0) or 0),
                "missing_lines": int(summary.get("missing_lines", 0) or 0),
            }
        )
    out.sort(key=lambda x: (-(x.get("num_statements") or 0), x.get("path") or ""))
    return out


def _group_vulture_by_file(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        grouped.setdefault(path, []).append(item)
    for path, arr in grouped.items():
        arr.sort(key=lambda x: (int(x.get("line") or 0), -(int(x.get("confidence") or 0))))
    return grouped


def _write_markdown_summary(out_dir: Path, report: Dict[str, Any]) -> None:
    v_items = ((report.get("vulture") or {}).get("items") or []) if isinstance(report.get("vulture"), dict) else []
    cov = (report.get("coverage") or {}).get("coverage") if isinstance(report.get("coverage"), dict) else None
    zero_files = _extract_zero_coverage_files(cov)
    grouped = _group_vulture_by_file(v_items if isinstance(v_items, list) else [])

    lines: List[str] = []
    lines.append("# Unused code audit report")
    lines.append("")
    lines.append("This report is generated by `scripts/unused_code_audit.py`.")
    lines.append("")

    lines.append("## 1) Coverage 0% files (review candidates)")
    lines.append("")
    if not zero_files:
        lines.append("- (none)")
    else:
        for item in zero_files[:200]:
            lines.append(f"- `{item['path']}` (statements={item['num_statements']})")
        if len(zero_files) > 200:
            lines.append(f"- ... and {len(zero_files) - 200} more")

    lines.append("")
    lines.append("## 2) Vulture candidates (static unused suspects)")
    lines.append("")
    if not grouped:
        lines.append("- (none)")
    else:
        shown_files = 0
        for path, items in sorted(grouped.items()):
            if shown_files >= 50:
                break
            shown_files += 1
            lines.append(f"- `{path}`")
            for it in items[:10]:
                msg = str(it.get("message") or "").strip()
                conf = int(it.get("confidence") or 0)
                line_no = int(it.get("line") or 0)
                lines.append(f"  - L{line_no} ({conf}%): {msg}")
            if len(items) > 10:
                lines.append(f"  - ... and {len(items) - 10} more")

    lines.append("")
    lines.append("## 3) Next step (2-stage removal)")
    lines.append("")
    lines.append("- Stage 1: Add candidates to `docs/DEPRECATIONS.md` and remove from public entrypoints (CLI / exports).")
    lines.append("- Stage 2: After a release (and audit re-run), delete files confirmed unused (0% coverage + no references).")
    lines.append("")

    _write_text(out_dir / "unused_code_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate vulture+coverage audit for unused code candidates.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for reports.")
    parser.add_argument("--package", type=str, default="automl_lib", help="Python package to analyze.")
    parser.add_argument("--min-confidence", type=int, default=80, help="Vulture min confidence (0-100).")
    parser.add_argument(
        "--coverage-cmd",
        action="append",
        default=[],
        help=(
            "Additional python args to run under coverage. Example: "
            "--coverage-cmd \"-m unittest -q\" or --coverage-cmd \"scripts/coverage_entrypoints_smoke.py\". "
            "(Default already runs unittest -q)"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_out_dir(out_dir)

    # Default dynamic trace: unit tests (offline-safe).
    coverage_cmds: List[List[str]] = [["-m", "unittest", "-q"]]
    for raw in args.coverage_cmd or []:
        tokens = [t for t in shlex.split(str(raw)) if t]
        if tokens:
            coverage_cmds.append(tokens)

    report: Dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "vulture": run_vulture(out_dir=out_dir, package=str(args.package), min_confidence=int(args.min_confidence)),
        "coverage": run_coverage(out_dir=out_dir, coverage_commands=coverage_cmds),
    }

    # Derived summary
    cov = (report.get("coverage") or {}).get("coverage") if isinstance(report.get("coverage"), dict) else None
    report["derived"] = {
        "zero_coverage_files": _extract_zero_coverage_files(cov),
    }

    _write_json(out_dir / "unused_code_report.json", report)
    _write_markdown_summary(out_dir, report)

    print(f"Wrote reports under: {out_dir}")
    print(f"- {out_dir / 'unused_code_report.md'}")
    print(f"- {out_dir / 'unused_code_report.json'}")


if __name__ == "__main__":
    main()
