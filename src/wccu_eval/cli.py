"""Command-line entry points. No network is used unless --execute is supplied."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .common import load_json, save_json, validate_job
from .selection import POLICIES, select_records
from .experiment import evaluate, aggregate
from .transport import ResponseCache
from .statistics import paired_bootstrap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wccu-eval")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select", help="Select records without any model call")
    s.add_argument("--jobs", type=Path, required=True)
    s.add_argument("--policy", choices=POLICIES, default="bundle")
    s.add_argument("--budget", type=int, default=1024)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("run", help="Replay a response cache, or explicitly collect new responses")
    for option in ("jobs", "labels", "cache", "output"):
        s.add_argument("--" + option, type=Path, required=True)
    s.add_argument("--demos", type=Path, required=True, help="Exact UTF-8 demonstration text, obtained separately")
    s.add_argument("--model", required=True)
    s.add_argument("--seed", type=int, default=20260921)
    s.add_argument("--caps", type=int, nargs="+", default=[512, 1024, 2048])
    s.add_argument("--policies", choices=POLICIES, nargs="+", default=["rank", "singleton", "bundle", "full"])
    s.add_argument("--execute", action="store_true")
    s.add_argument("--endpoint", help="Complete chat-completions endpoint; never inferred")
    s.add_argument("--max-new-calls", type=int, default=0)
    s.add_argument("--request-options", type=Path, help="Optional transport fields, e.g. local-server chat_template_kwargs")
    s = sub.add_parser("summarize")
    s.add_argument("--rows", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("compare", help="Compare two ID-keyed outcome dictionaries")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    s.add_argument("--metric", default="em")
    s.add_argument("--ratio", action="store_true")
    s.add_argument("--seed", type=int, default=20260921)
    s.add_argument("--replicates", type=int, default=20000)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.output.exists():
            raise ValueError("Output already exists; choose a new path")
        if a.command == "select":
            jobs = load_json(a.jobs)
            result = [{"id": j["id"], **select_records(j, a.policy, a.budget)} for j in jobs]
        elif a.command == "run":
            jobs, labels = load_json(a.jobs), load_json(a.labels)
            for job in jobs:
                validate_job(job)
            call = ResponseCache(a.cache, a.model, a.seed, execute=a.execute,
                                 endpoint=a.endpoint, max_new_calls=a.max_new_calls,
                                 request_options=load_json(a.request_options) if a.request_options else None)
            result = evaluate(jobs, labels, call, a.demos.read_text(encoding="utf-8"),
                              policies=tuple(a.policies), budgets=tuple(a.caps))
        elif a.command == "summarize":
            result = aggregate(load_json(a.rows))
        else:
            result = paired_bootstrap(load_json(a.a), load_json(a.b), a.metric, ratio=a.ratio,
                                      seed=a.seed, replicates=a.replicates)
        save_json(a.output, result)
        print(f"Wrote {a.output}")
        return 0
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
