"""Fixtures for scripts/bench-compaction.py.

Deliberately NOT under testfixtures/: that directory holds Dockerfiles and
fixture SERVERS for the test stacks, and every file in it is build context
for an image. These are importable Python modules the benchmark reads, and
nothing in the test suites may depend on them - a fixture whose shape is
tuned for a performance measurement is the wrong fixture for a correctness
assertion, and the two must not drift into each other.
"""
