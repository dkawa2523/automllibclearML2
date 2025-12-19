#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "automl_lib"


def _is_pkg_init(path: Path) -> bool:
    return path.name == "__init__.py"


def _path_to_module(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_package(module: str, *, is_package_init: bool) -> str:
    if is_package_init:
        return module
    if "." not in module:
        return module
    return module.rsplit(".", 1)[0]


def _resolve_relative_module(current_module: str, *, is_pkg_init: bool, level: int, module: Optional[str]) -> str:
    if level <= 0:
        return str(module or "")
    pkg = _module_package(current_module, is_package_init=is_pkg_init)
    parts = pkg.split(".") if pkg else []
    # level=1 means "from .", i.e. current package.
    up = level - 1
    if up:
        parts = parts[:-up] if up <= len(parts) else []
    if module:
        parts.extend(str(module).split("."))
    return ".".join([p for p in parts if p])


def _extract_imports(tree: ast.AST, *, current_module: str, is_pkg_init: bool) -> Set[str]:
    deps: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                name = str(alias.name or "").strip()
                if name:
                    deps.add(name)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            base = _resolve_relative_module(
                current_module,
                is_pkg_init=is_pkg_init,
                level=int(getattr(node, "level", 0) or 0),
                module=getattr(node, "module", None),
            ).strip()
            if base:
                deps.add(base)
                for alias in node.names:
                    if not alias.name or alias.name == "*":
                        continue
                    cand = f"{base}.{alias.name}"
                    deps.add(cand)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Handle importlib.import_module("automl_lib....") / __import__("automl_lib....")
            try:
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                if func_name not in {"importlib.import_module", "__import__"}:
                    return
                if not node.args:
                    return
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    name = arg0.value.strip()
                    if name:
                        deps.add(name)
            finally:
                self.generic_visit(node)

    Visitor().visit(tree)
    return deps


@dataclass(frozen=True)
class ModuleGraph:
    module_to_path: Dict[str, Path]
    edges: Dict[str, Set[str]]

    def reachable_from(self, roots: Iterable[str]) -> Set[str]:
        q: List[str] = []
        seen: Set[str] = set()
        for r in roots:
            if r in self.module_to_path and r not in seen:
                seen.add(r)
                q.append(r)
        while q:
            cur = q.pop(0)
            for dep in self.edges.get(cur, set()):
                # Only traverse within this repo's Python modules
                if dep not in self.module_to_path:
                    continue
                if dep in seen:
                    continue
                seen.add(dep)
                q.append(dep)
        # Importing a submodule implies importing all parent packages.
        expanded: Set[str] = set(seen)
        for mod in list(seen):
            parts = mod.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent in self.module_to_path:
                    expanded.add(parent)
        return expanded


def build_module_graph() -> ModuleGraph:
    module_to_path: Dict[str, Path] = {}
    edges: Dict[str, Set[str]] = {}

    for path in PKG_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        mod = _path_to_module(path)
        module_to_path[mod] = path

    for mod, path in module_to_path.items():
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue
        is_pkg_init = _is_pkg_init(path)
        deps = _extract_imports(tree, current_module=mod, is_pkg_init=is_pkg_init)
        # Keep only deps that are within automl_lib; external libs are ignored for reachability.
        deps_local = {d for d in deps if d == "automl_lib" or d.startswith("automl_lib.")}
        # Normalize deps to the nearest module we know about (package imports).
        normalized: Set[str] = set()
        for d in deps_local:
            cand = d
            while cand and cand not in module_to_path:
                if "." not in cand:
                    cand = ""
                    break
                cand = cand.rsplit(".", 1)[0]
            if cand:
                normalized.add(cand)
        edges[mod] = normalized

    return ModuleGraph(module_to_path=module_to_path, edges=edges)


def main() -> None:
    parser = argparse.ArgumentParser(description="Static import reachability report for automl_lib (offline cleanup aid).")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "automl_lib.cli.run_pipeline",
            "automl_lib.cli.run_preprocessing",
            "automl_lib.cli.run_training",
            "automl_lib.cli.run_inference",
            "automl_lib.cli.run_data_registration",
            "automl_lib.cli.run_data_editing",
            "automl_lib.cli.run_clone",
        ],
        help="Root modules treated as public entrypoints.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write report as JSON.")
    args = parser.parse_args()

    g = build_module_graph()
    roots = [str(r).strip() for r in (args.roots or []) if str(r).strip()]
    reachable = g.reachable_from(roots)
    all_mods = set(g.module_to_path.keys())
    unreachable = sorted(all_mods - reachable)

    # Do not recommend deleting plugin examples automatically (extension points).
    excluded_prefixes = ("automl_lib.plugins",)
    unreachable_recommend = [m for m in unreachable if not m.startswith(excluded_prefixes)]

    report = {
        "roots": roots,
        "reachable_count": len(reachable),
        "total_modules": len(all_mods),
        "unreachable_modules": unreachable,
        "unreachable_recommend_for_review": unreachable_recommend,
    }

    print(f"Reachable modules: {len(reachable)}/{len(all_mods)}")
    print(f"Unreachable modules: {len(unreachable)}")
    if unreachable_recommend:
        print("\nUnreachable (review candidates):")
        for m in unreachable_recommend:
            print(f" - {m}")

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report: {args.json}")


if __name__ == "__main__":
    main()
