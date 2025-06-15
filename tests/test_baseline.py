import re
import subprocess

def test_baseline_bench_runs_and_outputs_latency():
    # Run only 1 iteration on CPU to keep CI fast and stable
    result = subprocess.run(
        ["python3", "scripts/baseline_bench.py", "--iters", "1", "--device", "cpu"],
        capture_output=True,
        text=True,
        check=True
    )

    # Extract stdout lines
    lines = result.stdout.strip().splitlines()

    # Filter only the lines that start with our latency prefix
    lat_lines = [l for l in lines if l.strip().startswith("FP32 latency:")]
    assert lat_lines, f"No 'FP32 latency:' line found in output:\n{result.stdout}"

    # Take the last latency line
    lat_line = lat_lines[-1].strip()

    # Match "FP32 latency: 123.45 ms on cpu"
    m = re.match(r"FP32 latency:\s*([0-9]+\.[0-9]+)\s*ms on (cpu|mps|cuda)", lat_line)
    assert m, f"Invalid latency line format: {lat_line}"

    # Ensure the latency number is a positive float
    latency = float(m.group(1))
    assert latency > 0, "Latency must be positive"
