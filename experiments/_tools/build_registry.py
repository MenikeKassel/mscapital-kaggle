#!/usr/bin/env python
"""Compatibility entry point for the schema-v3 SSOT generator.

The old builder used schema v2 and is intentionally replaced by the explicit
one-way migration plus deterministic view generator.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent

def main():
    runpy.run_path(str(ROOT / "migrate_ssot_v3.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "build_project_views.py"), run_name="__main__")

if __name__ == "__main__": main()
