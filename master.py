import subprocess
import time
import tracemalloc
import json
import os
from pathlib import Path
import subprocess
import time
import json
import threading
from pathlib import Path

SYSTEMS = {
    "mediapipe": "mediapipe_benchmark.py",
    "tflite":    "tflite_benchmark.py",
    "roboflow":  "roboflow_benchmark.py",
}

GROUND_TRUTH_MAP = {
    "fist":      "fist",
    "palm":      "open_palm",
    "peace":     "peace",
    "point":     "pointing",
    "thumbs_up": "thumbs_up",
}

def get_ground_truth(filename):
    stem = Path(filename).stem  # e.g. "fist1"
    for key, label in GROUND_TRUTH_MAP.items():
        if stem.startswith(key):
            return label
    return "unknown"

def get_peak_rss(pid):
    """Read current RSS in KB from /proc/PID/status"""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # in KB
    except FileNotFoundError:
        return 0
    return 0

def monitor_memory(pid, interval=0.01):
    """Poll child process memory in a background thread, return peak KB"""
    peak = 0
    while True:
        rss = get_peak_rss(pid)
        if rss == 0:
            break
        peak = max(peak, rss)
        time.sleep(interval)
    return peak

def run_single(script, image_path):
    proc = subprocess.Popen(
        ["python3", script, image_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Monitor memory in background thread while process runs
    peak_rss = [0]
    def _monitor():
        peak_rss[0] = monitor_memory(proc.pid)
    
    monitor_thread = threading.Thread(target=_monitor)
    monitor_thread.start()

    start = time.perf_counter()
    stdout, stderr = proc.communicate()  # waits for process to finish
    elapsed = time.perf_counter() - start

    monitor_thread.join()

    try:
        output = json.loads(stdout.strip().splitlines()[-1])
        gesture = output.get("gesture", "unknown")
    except Exception:
        gesture = "error"
        print(f"  [parse error] stdout: {stdout!r} stderr: {stderr!r}")

    return gesture, elapsed, peak_rss[0]

images = sorted(Path("images").glob("*.jpeg"))

# results[system][image] = {gesture, correct, time, memory}
results = {name: [] for name in SYSTEMS}

for image_path in images:
    ground_truth = get_ground_truth(image_path.name)
    print(f"\n{image_path.name} (expected: {ground_truth})")

    for system_name, script in SYSTEMS.items():
        gesture, elapsed, peak_mb = run_single(script, str(image_path))
        correct = gesture == ground_truth
        results[system_name].append({
            "image":    image_path.name,
            "expected": ground_truth,
            "got":      gesture,
            "correct":  correct,
            "time_ms":  round(elapsed * 1000, 1),
            "mem_kb": peak_mb,
        })
        status = "✓" if correct else "✗"
        print(f"  {status} {system_name:12} → {gesture:12} | {elapsed*1000:.1f}ms | {peak_mb/1000:.1f}MB")

# Summary
print("\n" + "="*60)
print(f"{'SYSTEM':<14} {'ACCURACY':>10} {'AVG TIME':>10} {'AVG MEM':>10}")
print("="*60)
for system_name, records in results.items():
    accuracy = sum(r["correct"] for r in records) / len(records) * 100
    avg_time = sum(r["time_ms"] for r in records) / len(records)
    avg_mem = sum(r["mem_kb"] for r in records) / len(records)
    print(f"{system_name:<14} {accuracy:>9.1f}% {avg_time:>9.1f}ms {avg_mem/1000:>9.1f}MB")

# Save full results to JSON
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nFull results saved to benchmark_results.json")