import re
import subprocess

def test_baseline_bench_runs_and_outputs_latency():
    # Run only 1 iteration to keep CI fast
    result = subprocess.run(
        ["python3", "scripts/baseline_bench.py", "--iters", "1"],
        capture_output=True, text=True, check=True
    )
    out = result.stdout.strip()
    # Expect something like "FP32 latency: 26.24 ms on mps" or "on cuda"
    match = re.match(r"FP32 latency:\s+(\d+\.\d+)\s+ms on (mps|cuda|cpu)", out)
    assert match, f"Unexpected output from baseline_bench.py: {out}"
    # And the number must be > 0
    latency = float(match.group(1))
    assert latency > 0
