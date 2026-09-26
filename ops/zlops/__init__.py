"""zlops — the `zl` pod-ops bundle.

One command per repeated operator action (see ops/README.md). This package
only ORCHESTRATES the scripts already in `scripts/` and `compactor/`: it
picks the right interpreter, stops/starts the right services, confirms
before it changes anything, and logs what happened. It never re-implements
a repair script's own logic -- two copies of one rule drifting apart is
this codebase's most expensive recurring defect (see memory.md).
"""
