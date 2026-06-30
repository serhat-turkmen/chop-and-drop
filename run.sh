#!/usr/bin/env bash
# Launch Chop & Drop from the repo directory, regardless of where it's called from.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/chop_and_drop.py" "$@"
