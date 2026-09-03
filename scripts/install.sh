#!/usr/bin/env bash
set -euo pipefail

# Kept as a convenience wrapper. `uv sync` is the canonical installation path.
uv sync "$@"
