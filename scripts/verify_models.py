#!/usr/bin/env python3
"""Developer convenience wrapper around the public model verifier."""

from core_retarget.cli.main import main

raise SystemExit(main(["robots", "verify"]))
