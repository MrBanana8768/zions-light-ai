"""
Every compactor module that the image needs must actually be COPYed into it.

WHY THIS FILE EXISTS. This defect has now shipped three times in three
consecutive releases, and each time the missing line was for a module added
in the same change that needed it:

  * v3.1.3 added `tokenhealth.py`. v3.1.4 shipped without the COPY line, and
    the compactor went FATAL on boot in production — "ModuleNotFoundError:
    No module named 'tokenhealth'" — with the chat path down until an
    operator hot-copied the file into the running container.
  * v3.2 added `dbselect.py`. Caught during review, before it shipped.
  * v3.1.7 added `tailhealth.py`, imported at main.py module scope and again
    by health.py. Caught by a cold reviewer, before it shipped.

The Dockerfile grew two BUILD GUARDs in response, and they work — but they
only fail during `docker build`, which on this project means after a tag has
been cut, on a machine with a GPU base image, twenty minutes into a layer
cache miss. The information needed to catch it is entirely static and sits
in two files. This test is the same guard moved to where it costs seconds.

WHAT IT CHECKS. Starting from the entry points the image actually executes
(main via uvicorn, and the scripts supervisord and entrypoint.sh invoke), it
follows `import X` / `from X import ...` transitively through the local
modules and asserts every one of them appears in a `COPY compactor/X.py`
line in the Dockerfile.

Static, not dynamic: it reads the source with `ast` rather than importing
anything. An import inside a `try:` or a function body counts exactly the
same as a top-level one, because `import tailhealth` inside
`gather_health_full` is just as fatal at the moment it runs — health.py does
precisely that, and a test that only looked at module-scope imports would
have missed it.

    python test_image_manifest.py
"""

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCKERFILE = HERE.parent / "Dockerfile"

# What the running image actually starts. main.py is the compactor itself;
# the rest are invoked by supervisord or entrypoint.sh as subprocesses, which
# is why a module only THEY need is just as required as one main needs.
ENTRY_POINTS = ("main", "selftest", "backup")

FAILED = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILED.append(label)


def local_modules() -> set[str]:
    """Every non-test module beside this file — the candidates for import."""
    return {
        p.stem
        for p in HERE.glob("*.py")
        if not p.name.startswith("test_")
    }


def imports_of(module: str, known: set[str]) -> set[str]:
    """Local modules `module` imports, at ANY nesting depth.

    ast.walk, not a scan of `tree.body`: health.py imports tailhealth inside
    a function, and that import is exactly as load-bearing as a top-level one
    — it just fails later, on a request instead of at boot, which is worse.
    """
    src = (HERE / f"{module}.py").read_text(encoding="utf-8")
    found: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in known:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has no module; level > 0 is a relative import.
            if node.module:
                root = node.module.split(".")[0]
                if root in known:
                    found.add(root)
            if node.level:
                for a in node.names:
                    if a.name in known:
                        found.add(a.name)
    return found


def required_modules() -> dict[str, list[str]]:
    """Transitive closure from the entry points -> the chain that reached it.

    The chain is kept so a failure can say WHY a module is needed, which is
    the first thing anyone asks when a build guard trips.
    """
    known = local_modules()
    need: dict[str, list[str]] = {}
    stack = [(e, [e]) for e in ENTRY_POINTS if (HERE / f"{e}.py").exists()]
    while stack:
        mod, chain = stack.pop()
        if mod in need:
            continue
        need[mod] = chain
        for dep in sorted(imports_of(mod, known)):
            if dep not in need:
                stack.append((dep, chain + [dep]))
    return need


def copied_modules() -> set[str]:
    """Module names on `COPY compactor/<name>.py ...` lines in the Dockerfile."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    return set(re.findall(r"^COPY\s+compactor/([A-Za-z0-9_-]+)\.py\s", text, re.M))


print("\n[1] the Dockerfile COPYs every module the image needs")
need = required_modules()
copied = copied_modules()
check(len(copied) > 10, f"parsed the Dockerfile COPY list ({len(copied)} modules)")
check("main" in need, "walked the import graph from the entry points")

missing = sorted(set(need) - copied)
for m in missing:
    print(f"       MISSING: compactor/{m}.py  (reached via {' -> '.join(need[m])})")
check(
    not missing,
    f"all {len(need)} required module(s) are COPYed"
    + (f" — missing {missing}" if missing else ""),
)

print("\n[2] the guard has teeth")
# If this file cannot detect a module that IS absent, [1] proves nothing. Drop
# a real entry from the parsed COPY set and confirm the same comparison fails.
_pretend = copied - {"memory"}
check(
    "memory" in (set(need) - _pretend),
    "removing a COPY line from the parsed set is detected as missing",
)
# And the import walk must actually see a nested import. health.py imports
# tailhealth inside a function; that is the shape v3.1.7 nearly shipped.
if (HERE / "health.py").exists() and "tailhealth" in local_modules():
    check(
        "tailhealth" in imports_of("health", local_modules()),
        "a function-scoped `import tailhealth` in health.py is seen",
    )

print("\n[3] nothing is COPYed that does not exist")
# The other direction: a COPY line for a deleted module fails `docker build`
# with a much less obvious error than a missing one.
ghosts = sorted(m for m in copied if not (HERE / f"{m}.py").exists())
check(not ghosts, f"every COPYed module exists on disk{f' — ghosts {ghosts}' if ghosts else ''}")

print()
if FAILED:
    for f in FAILED:
        print("FAIL " + f)
    sys.exit(1)
print("All image-manifest tests passed.")
