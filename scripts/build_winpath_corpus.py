#!/usr/bin/env python3
"""Generate a synthetic corpus of Windows-path-heavy text for KLD/perplexity eval.

Motivation: the standard KLD eval corpus (wikitext-2) is forward-slash/prose text and never
exercises backslash-escape-collision positions (\\t, \\r, \\n, \\b, \\f as *path separator +
first letter of a directory/file name*, not as the string escape they spell in every other
context). A real corruption was observed where a quantized/sampled model emitted a literal LF
byte at exactly one of those positions inside a tool-call JSON `file_path` argument -- see
data/logs/prompts/20260817_194601_451084.json. wikitext contains ~zero examples of this pattern,
so a KLD sweep run only against it cannot detect a regression in this specific failure mode.

This generates agent-transcript-shaped text (tool-call JSON blocks, prose describing paths,
directory listings) with realistic Windows paths across common directory shapes (user profile,
AppData, Program Files, System32, project/job dirs, node_modules, temp), biased toward directory
and file name starts that collide with JSON escape letters (t, r, n, b, f) without excluding
other letters -- the goal is realistic distribution, not a pathological one-note corpus.

Deterministic (fixed seed) so re-runs are reproducible/diffable.

Usage:
  python scripts/build_winpath_corpus.py <output.txt> [--target-bytes N] [--seed N]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DRIVES = ["C:", "D:", "H:"]

USERS = ["zacaj", "alice", "bcarter", "dev", "svc-build"]

# Directory name pools, deliberately split into "collision-prone" (starts with t/r/n/b/f, the
# letters JSON/C escape sequences use after backslash) and "other" so generated paths cover both
# without being all-collision.
DIRS_COLLISION = [
    "tmp", "temp", "tools", "Templates", "test", "target",
    "repair", "reports", "release", "run", "results",
    "node_modules", "new", "notes",
    "bin", "backup", "build",
    "fixtures", "flake", "final",
]
DIRS_OTHER = [
    "Users", "AppData", "Roaming", "Local", "Documents", "Desktop", "Downloads",
    "Program Files", "Program Files (x86)", "Windows", "System32", "ProgramData",
    "src", "scripts", "ComfyUI", "output", "models", "config", "logs", "cache",
    "SwarmUI", "workspace", "projects", "vendor", "packages", "dist",
]
FILES_COLLISION = [
    "repair.py", "test.py", "table.csv", "report.json", "readme.md", "run.sh",
    "boot.log", "format.txt", "final.gguf", "notes.txt", "new_model.bin",
    "temp.tsv", "tools.json", "results.md", "build.log",
]
FILES_OTHER = [
    "config.yaml", "index.js", "main.py", "server.py", "package.json",
    "presets.json", "chat_template.jinja", "requirements.txt", "app.py",
    "comfy_helper.py", "settings.json", "state.json", "timeline.jsonl",
    "gen.py", "dump.ps1", "inspect.py", "model.safetensors",
]

JOB_HEX = lambda rng: "".join(rng.choice("0123456789abcdef") for _ in range(8))


def rand_component(rng: random.Random) -> str:
    pool = rng.choice([DIRS_COLLISION, DIRS_OTHER])
    return rng.choice(pool)


def rand_path(rng: random.Random, depth: int | None = None) -> str:
    drive = rng.choice(DRIVES)
    user = rng.choice(USERS)
    depth = depth if depth is not None else rng.randint(2, 5)

    parts = [drive + "\\", "Users", user]
    for _ in range(depth):
        parts.append(rand_component(rng))
    if rng.random() < 0.3:
        parts.append(".claude")
        parts.append("jobs")
        parts.append(JOB_HEX(rng))
    file_pool = rng.choice([FILES_COLLISION, FILES_OTHER])
    parts.append(rng.choice(file_pool))

    # first element already ends in a backslash (drive root); join the rest with backslashes
    return parts[0] + "\\".join(parts[1:])


TOOLS = ["Write", "Edit", "Read", "PowerShell", "Bash"]

PROSE_TEMPLATES = [
    "The build artifacts were written to {p}, which the packaging step reads back before "
    "zipping the release.",
    "I checked {p} and confirmed the config still points at the old server address.",
    "Logs for the failed run are under {p} -- tail that file if the job dies again.",
    "The installer places its runtime under {p}, separate from the per-user data in {p2}.",
    "Deleting {p} did not help; the real state was cached at {p2}.",
    "Our CI agent stages the checkout at {p} before running the test suite.",
    "The crash dump landed in {p}, about 40MB, mostly stack frames from the render thread.",
    "Model weights are expected at {p}; if that path is missing the loader falls back to {p2}.",
    "The script backs up the previous config to {p} before overwriting it.",
    "Permissions on {p} were wrong -- the service account couldn't write there.",
]

TOOL_CALL_TEMPLATE = '''{{
  "type": "tool_use",
  "name": "{tool}",
  "input": {{
    "file_path": "{p}"{extra}
  }}
}}'''

RESULT_TEMPLATES = [
    "File created successfully at: {p}",
    "File does not exist. Note: your current working directory is {p2}.",
    '''Get-ChildItem -Path '{p}' -Force | Select-Object Name,Length,LastWriteTime''',
    "1\t{p}\n2\t# generated by build step\n",
]


def gen_block(rng: random.Random) -> str:
    p = rand_path(rng)
    p2 = rand_path(rng)
    kind = rng.random()
    if kind < 0.4:
        return rng.choice(PROSE_TEMPLATES).format(p=p, p2=p2)
    elif kind < 0.75:
        tool = rng.choice(TOOLS)
        extra = ""
        if tool == "Edit":
            extra = ',\n    "old_string": "x = 1",\n    "new_string": "x = 2"'
        elif tool == "PowerShell":
            extra = ',\n    "command": "Get-ChildItem -Path \'%s\'"' % p.replace("\\", "\\\\")
        return TOOL_CALL_TEMPLATE.format(tool=tool, p=p.replace("\\", "\\\\"), extra=extra)
    else:
        return rng.choice(RESULT_TEMPLATES).format(p=p, p2=p2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-bytes", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; use --force to overwrite")

    rng = random.Random(args.seed)
    blocks = []
    total = 0
    while total < args.target_bytes:
        b = gen_block(rng)
        blocks.append(b)
        total += len(b) + 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        out.write("\n\n".join(blocks))
        out.write("\n")

    print(f"{len(blocks)} blocks, {total/1e3:.1f}KB -> {args.output}")


if __name__ == "__main__":
    main()
