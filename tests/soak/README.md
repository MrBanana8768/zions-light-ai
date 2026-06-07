# Soak watch (V2.3 Theme 3)

Long-running leak watch for the compactor: samples resident memory (RSS)
and open file-descriptor count over time and flags slow monotonic growth.
The leak suspects in this stack are unclosed `httpx` clients, ChromaDB
handles, and the background-task set (now bounded by `bgwork` — the soak
also confirms that bound actually holds under sustained load).

Pod-local + Linux-only (reads `/proc`). It's a monitoring tool, not a
pytest — run it on the pod, ideally for hours to days.

## Running

On the pod, using the compactor venv (the `--drive` option needs `httpx`,
which that venv has):

```bash
# Quick 1-hour check, generating its own load
/opt/compactor-venv/bin/python /data/zions-src/tests/soak/soak_monitor.py \
    --duration-hours 1 --drive

# A real multi-day soak, sampling every 5 min, logging to a file
/opt/compactor-venv/bin/python /data/zions-src/tests/soak/soak_monitor.py \
    --duration-hours 72 --interval-s 300 --drive --out /data/soak.jsonl
```

(`tests/` isn't baked into the image — clone the repo onto the pod first,
same as the chaos + integration suites.)

Run it in the background (e.g. `nohup ... &` or a tmux pane) for long soaks;
it flushes each sample to `--out` so a disconnect doesn't lose data.

## What it reports

Per sample: `t=<hours>  rss=<MB>  fd=<count>`. At the end, a summary with the
first→last delta and the least-squares slope (MB/h and FD/h):

```
=== SOAK SUMMARY ===
  RSS: 540.2 → 548.9 MB  (Δ +8.7 MB, slope +0.4 MB/h)
  FD:  41 → 42  (Δ +1, slope +0.0/h)
  ✓ stable — no leak signature.
```

## Pass / fail

- **Exit 0** — stable. RSS may wobble (caches, allocator) but isn't trending
  up past the threshold, and FD count is flat.
- **Exit 1** — suspected leak: RSS grew more than `ZIONS_SOAK_RSS_GROWTH_MB`
  (default 150 MB) with a positive slope, and/or FDs grew more than
  `ZIONS_SOAK_FD_GROWTH` (default 50) with a positive slope.
- **Exit 2** — couldn't find the compactor process (pass `--pid`, or run on
  the pod).

Tune thresholds via `ZIONS_SOAK_RSS_GROWTH_MB` / `ZIONS_SOAK_FD_GROWTH`.

## Interpreting a flag

A positive RSS slope over a long window with steadily-climbing FDs is the
classic leak signature. If you see it:
- check `/health/full` → `background_work.outstanding` — is it climbing
  (tasks not completing) or stable?
- `ls /proc/<pid>/fd | wc -l` and look for many sockets → httpx clients not
  being closed.
- A flat FD count with rising RSS points at a Python-object leak (e.g. an
  ever-growing dict/list) rather than a handle leak.

This is the V2.3 "failure-tested over real time" item: it isn't "done" until
a multi-day soak has actually been run and come back stable.
