"""Local gate: run N JevBench tasks on ONE model through app.answer() via bench.start_run, exactly the
code path the Space's Run button uses. Usage: PYTHONPATH=<space root> python -m bench.local_run [--model NAME] [--n 20]
Writes bench/runs/<run_id>/ and pushes to the private dataset when HF_TOKEN is set."""
import argparse, os, sys, time, json

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--tier", default=None)
a = ap.parse_args()

os.environ.setdefault("DECIDER_PUBLIC_ONLY", "1")
import app  # noqa: E402
import bench  # noqa: E402

app.load_models()
names = app.model_choices()
model = a.model or next((n for n in names if "2B-qwen3.5" in n), names[0])
print("model", model, "device", app.runner.default_device(), flush=True)

# a 20-task subset spread over tiers/families: every k-th task in board order
ids = bench.task_ids(a.tier)
step = max(1, len(ids) // a.n)
subset = ids[::step][: a.n]
bench.task_ids = lambda tier_filter=None, _s=subset: list(_s)  # the run reads its ids through this hook
secs, src = bench.estimate_s([model], len(subset), app.runner.default_device())
print(f"estimate {secs:.0f}s ({src})", flush=True)
job = bench.start_run(app.answer, [model], None, label=f"local {len(subset)}-task subset · {model}")
assert not isinstance(job, str), job
print("run", job.run_id, "dir", job.dir, flush=True)
t0 = time.time()
while job.running():
    time.sleep(2)
    print(f"  {job.done}/{job.total} done · {time.time() - t0:.0f}s · eta {job.eta_s()}", flush=True)
job.thread.join()
print(json.dumps(job.meta(), indent=1, default=str))
