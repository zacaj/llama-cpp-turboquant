#!/usr/bin/env python3
"""Reproduce and sampler-sweep the Windows-path escape-collision corruption.

Context: a real incident (data/logs/prompts/20260817_194601_451084.json) showed a raw 0x0A byte
sampled directly after a backslash inside a JSON tool-call `file_path` argument, at a position
where the correct continuation was the letter 't' (i.e. the model emitted a literal newline where
it needed to write `\\t` as two separate characters of a Windows path -- `\tmp`). The hypothesis is
that `\`+{t,r,n,b,f} is an escape-collision position: in the vast majority of training text,
backslash followed by one of those letters spells a control-character escape, not a literal path
component, so under a smeared/low-confidence quantized distribution the model occasionally samples
the escape instead of the literal letter (writing a literal newline where it needed
backslash-t, two separate characters of a Windows path).

This script asks the live server (must be running, see docker-compose.yml service `llama-cpp`) to
copy many distinct Windows paths verbatim into a forced tool call, across a grid of sampler
settings, and measures how often the returned argument doesn't byte-for-byte match what was asked
for -- specifically flagging raw C0 control bytes (0x00-0x1F) landing right after a backslash,
which is the exact corruption signature observed in the incident.

Talks to the OpenAI-compatible /v1/chat/completions endpoint (not /v1/messages) because the
Anthropic-compat layer does not forward `min_p` at all (see server-chat.cpp
server_chat_convert_anthropic_to_oai, the passthrough whitelist is only
temperature/top_p/top_k/stream/chat_template_kwargs) -- both endpoints route into the same core
sampling/grammar/tool-call code, so this loses no fidelity for what's under test here.

Usage:
  python3 scripts/winpath_corruption_probe.py --trials 40 --configs prod,fix
  python3 scripts/winpath_corruption_probe.py --trials 40 --configs prod,fix,minp-only,temp-only,required
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
import urllib.error

SERVER = "http://localhost:3003/v1/chat/completions"

DRIVES = ["C:", "D:", "H:"]
USERS = ["zacaj", "alice", "bcarter"]
DIRS_COLLISION = ["tmp", "temp", "tools", "test", "target", "repair", "reports",
                   "run", "results", "new", "notes", "bin", "backup", "fixtures"]
DIRS_OTHER = ["Users", "AppData", "Roaming", "Local", "Documents", "src", "scripts",
              "ComfyUI", "output", "models", "config", "SwarmUI", "workspace"]
FILES_COLLISION = ["repair.py", "test.py", "table.csv", "report.json", "run.sh",
                    "format.txt", "notes.txt", "temp.tsv", "tools.json", "build.log"]
FILES_OTHER = ["config.yaml", "index.js", "main.py", "server.py", "package.json",
               "presets.json", "settings.json", "state.json", "gen.py"]

JOB_HEX = lambda rng: "".join(rng.choice("0123456789abcdef") for _ in range(8))


def rand_path(rng: random.Random) -> str:
    drive = rng.choice(DRIVES)
    user = rng.choice(USERS)
    parts = [drive + "\\", "Users", user]
    for _ in range(rng.randint(2, 4)):
        parts.append(rng.choice(DIRS_COLLISION if rng.random() < 0.6 else DIRS_OTHER))
    if rng.random() < 0.4:
        parts += [".claude", "jobs", JOB_HEX(rng)]
    parts.append(rng.choice(FILES_COLLISION if rng.random() < 0.6 else FILES_OTHER))
    return parts[0] + "\\".join(parts[1:])


TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute Windows path to write to."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": ["file_path", "content"],
        },
    },
}]

CTRL_AFTER_BACKSLASH_RE = re.compile(r"\\[\x00-\x1f]")


def make_request_body(path: str, sampler: dict, tool_choice: str, reasoning_effort: str) -> dict:
    return {
        "model": "Qwen3.6-35B-A3B",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call the write_file tool. Set file_path to EXACTLY this string, "
                    "copied byte-for-byte with no changes whatsoever "
                    "(it is a Windows path, copy every backslash literally, do not interpret "
                    "any part of it as an escape sequence):\n\n" + path +
                    "\n\nSet content to the single word \"ok\"."
                ),
            }
        ],
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "max_tokens": 300,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
        **sampler,
    }


def call_server(body: dict, timeout: float = 120.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(SERVER, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def extract_file_path(resp: dict) -> str | None:
    try:
        choice = resp["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == "write_file":
                args = json.loads(tc["function"]["arguments"])
                return args.get("file_path")
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        return None
    return None


CONFIGS = {
    # name: (sampler dict, tool_choice, reasoning_effort, description)
    "prod":       (dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0),  "auto",     "low", "current production sampler"),
    "fix":        (dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.05), "auto",     "low", "proposed fix (temp 0.7, min_p 0.05)"),
    "minp-only":  (dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.05), "auto",     "low", "isolate min_p at prod temp"),
    "temp-only":  (dict(temperature=0.7, top_p=0.95, top_k=20, min_p=0.0),  "auto",     "low", "isolate temp at prod min_p"),
    "required":   (dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0),  "required", "low", "prod sampler, grammar forced from token 0"),
    "greedy":     (dict(temperature=0.0),                                                        "auto", "low", "greedy baseline sanity check"),
}


def run_config(name: str, n_trials: int, seed: int) -> dict:
    sampler, tool_choice, reasoning_effort, desc = CONFIGS[name]
    rng = random.Random(seed)
    n_ok = 0
    n_mismatch = 0
    n_ctrl_corrupt = 0
    n_no_call = 0
    n_error = 0
    examples = []
    for i in range(n_trials):
        path = rand_path(rng)
        body = make_request_body(path, sampler, tool_choice, reasoning_effort)
        try:
            resp = call_server(body)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            n_error += 1
            continue
        got = extract_file_path(resp)
        if got is None:
            n_no_call += 1
            continue
        if got == path:
            n_ok += 1
        else:
            n_mismatch += 1
            is_ctrl = bool(CTRL_AFTER_BACKSLASH_RE.search(got))
            if is_ctrl:
                n_ctrl_corrupt += 1
            if len(examples) < 8:
                examples.append({"expected": path, "got": got, "ctrl_after_backslash": is_ctrl})
        print(f"  [{name}] {i+1}/{n_trials}  ok={n_ok} mismatch={n_mismatch} ctrl={n_ctrl_corrupt} "
              f"no_call={n_no_call} err={n_error}", file=sys.stderr)
    return {
        "config": name, "desc": desc, "n_trials": n_trials,
        "n_ok": n_ok, "n_mismatch": n_mismatch, "n_ctrl_corrupt": n_ctrl_corrupt,
        "n_no_call": n_no_call, "n_error": n_error, "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--configs", type=str, default="prod,fix")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    names = args.configs.split(",")
    for n in names:
        if n not in CONFIGS:
            raise SystemExit(f"unknown config {n!r}, choose from {list(CONFIGS)}")

    results = []
    for name in names:
        t0 = time.time()
        r = run_config(name, args.trials, args.seed)
        r["wall_s"] = time.time() - t0
        results.append(r)

    print("\n\n=== SUMMARY ===")
    print(f"{'config':<12} {'desc':<42} {'mismatch':>9} {'ctrl-corrupt':>13} {'no_call':>8} {'err':>4} {'wall_s':>8}")
    for r in results:
        print(f"{r['config']:<12} {r['desc']:<42} "
              f"{r['n_mismatch']}/{r['n_trials']:<7} {r['n_ctrl_corrupt']}/{r['n_trials']:<11} "
              f"{r['n_no_call']:>8} {r['n_error']:>4} {r['wall_s']:>7.1f}s")

    for r in results:
        if r["examples"]:
            print(f"\n--- {r['config']} mismatch examples ---")
            for ex in r["examples"]:
                print(f"  ctrl_after_backslash={ex['ctrl_after_backslash']}")
                print(f"    expected: {ex['expected']!r}")
                print(f"    got:      {ex['got']!r}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
