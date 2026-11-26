"""Release gate for packaging: build the wheel and the source distribution, install each into a
clean virtual environment, move outside the checkout, and exercise the installed package.

    python -m benchmarks.release_check                 # build, install, smoke-test both artifacts
    python -m benchmarks.release_check --require-reference   # also run the transformers parity
                                                              # tests, which must not be skipped

The smoke test (``python -m benchmarks.release_check smoke``, run inside each environment) checks:
imports; that the bundled configuration files are the installed copy and all present; a tiny
real runtime operation (write a small GGUF model, run it, speculative decoding); and the
supported commands.  A check that cannot run fails the gate -- nothing is skipped silently.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

COMMANDS = ("basp-verify-runtime", "basp-run-study", "basp-speculative-bench", "basp-calibrate", "basp-explore",
            "basp-report", "basp-release-check")  # fmt: skip


def smoke(expected_configs: list[str]) -> dict[str, object]:
    """Runs inside the installed environment, outside the checkout."""
    import numpy as np

    import basp
    from basp.evidence import CONFIG_DIR, IN_CHECKOUT, runtime_fingerprint
    from basp.experiments.profiles import load_json
    from basp.models.gguf_writer import tiny_qwen3
    from basp.runtime import Engine, generate
    from basp.runtime.speculative import NgramDrafter, speculative_generate

    report: dict[str, object] = {"version": basp.__version__, "package": str(Path(basp.__file__).parent)}
    assert not IN_CHECKOUT, f"the installed package believes it runs from a checkout ({CONFIG_DIR})"
    assert "site-packages" in str(CONFIG_DIR) and CONFIG_DIR.is_dir(), f"configurations not installed: {CONFIG_DIR}"
    present = sorted(p.name for p in CONFIG_DIR.glob("*.json"))
    assert present == sorted(expected_configs), f"bundled configurations differ: {present}"
    for name in present:  # every one parses through the package's own loader
        load_json(name[:-5])
    report["configs"] = len(present)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.gguf"
        tiny_qwen3(path, moe=True, seed=3)
        engine = Engine(path, expert_cache_bytes=2 * 3 * 32 * 64 * 4)
        prompt = [5, 17, 3, 44, 9, 5, 17, 3]
        with engine.session(32) as s:
            plain = generate(s, prompt, 6).tokens
        with engine.session(32) as s:
            spec = speculative_generate(s, NgramDrafter(), prompt, 6, k=3).tokens
        assert plain == spec and len(plain) == 6, (plain, spec)
        engine.weights.check_invariant()
        assert np.isfinite(engine.memory_report()["expert_arrays"]["peak_resident_bytes"])
    report["runtime"] = runtime_fingerprint()[:12]
    for command in COMMANDS:
        done = subprocess.run([command, "--help"], capture_output=True, text=True, timeout=300, check=False)
        assert done.returncode == 0, f"{command} --help failed: {done.stderr[-400:]}"
    done = subprocess.run(["basp-run-study", "headroom", "--fraction", "0.5", "--reduction", "0.5"],
                          capture_output=True, text=True, timeout=120, check=False)  # fmt: skip
    assert done.returncode == 0 and "1.333x" in done.stdout, done.stdout + done.stderr
    report["commands"] = len(COMMANDS)
    return report


def _python(env: Path) -> Path:
    return env / ("Scripts" if os.name == "nt" else "bin") / "python"


def build(repo: Path, dist: Path) -> list[Path]:
    """Source distribution first; the wheel is built *from* it, so the sdist is proven complete."""
    if shutil.which("uv"):
        subprocess.run(["uv", "build", "--sdist", "--wheel", "--out-dir", str(dist), str(repo)], check=True,
                       capture_output=True, text=True)  # fmt: skip
    else:
        subprocess.run([sys.executable, "-m", "build", "--outdir", str(dist), str(repo)], check=True)
    artifacts = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    if len(artifacts) != 2:
        raise SystemExit(f"expected one wheel and one source distribution, found {[a.name for a in artifacts]}")
    return artifacts


def check(repo: Path, artifact: Path, work: Path, *, require_reference: bool) -> dict[str, object]:
    env = work / f"env-{artifact.suffix.strip('.')}"
    venv.EnvBuilder(with_pip=True, clear=True).create(env)
    python = _python(env)
    extras = "[dev,verify]" if require_reference else ""
    subprocess.run([str(python), "-m", "pip", "install", "-q", f"{artifact}{extras}"], check=True)
    configs = sorted(p.name for p in (repo / "configs").glob("*.json"))
    outside = work / "outside"
    outside.mkdir(exist_ok=True)
    env_vars = {**os.environ, "PATH": f"{python.parent}{os.pathsep}{os.environ['PATH']}"}
    env_vars.pop("PYTHONPATH", None)
    code = ("import json, sys; from benchmarks.release_check import smoke; "
            f"print(json.dumps(smoke({configs!r})))")  # fmt: skip
    done = subprocess.run([str(python), "-I", "-c", code], cwd=outside, env=env_vars, capture_output=True, text=True,
                          check=False)  # fmt: skip
    if done.returncode != 0:
        raise SystemExit(f"{artifact.name}: smoke test failed\n{done.stdout}\n{done.stderr[-3000:]}")
    result = json.loads(done.stdout.strip().splitlines()[-1])
    if require_reference:  # the transformers comparison is a required check here: a skip fails it
        tests = work / "tests"
        shutil.copytree(repo / "tests", tests, dirs_exist_ok=True)
        run = subprocess.run([str(python), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v04_verification.py"],
                             cwd=work, env={**env_vars, "BASP_REQUIRE_REFERENCE": "1"}, capture_output=True, text=True,
                             check=False)  # fmt: skip
        tail = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-500:]
        if run.returncode != 0 or "skipped" in tail:
            raise SystemExit(f"{artifact.name}: reference verification did not pass: {tail}")
        result["reference_verification"] = tail
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--require-reference", action="store_true",
                        help="install torch and transformers and run the parity tests; a skip fails the gate")  # fmt: skip
    parser.add_argument("--keep", type=Path, help="keep the build and the environments here")
    args = parser.parse_args(argv)
    if not (args.repo / "pyproject.toml").is_file():
        raise SystemExit(f"{args.repo} is not a source checkout: the release check builds from one")
    work = args.keep or Path(tempfile.mkdtemp(prefix="basp-release-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = build(args.repo, work / "dist")
        for artifact in artifacts:
            result = check(args.repo, artifact, work, require_reference=args.require_reference)
            print(f"{artifact.name}: OK {json.dumps(result)}")
    finally:
        if args.keep is None:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
