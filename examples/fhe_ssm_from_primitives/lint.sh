#!/usr/bin/env bash
# Lint, format and type-check this example. Fixes what the tools can fix, then fails on
# anything left. Tools are taken from $RUFF / $MYPY, else from PATH; mypy runs against the
# interpreter in $PYTHON (default: python3) so it sees that environment's numpy and perseus.
#
#   examples/fhe_ssm_from_primitives/lint.sh           fix + check
#   examples/fhe_ssm_from_primitives/lint.sh --check   check only (CI: changes nothing)
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
ruff=${RUFF:-ruff}
mypy=${MYPY:-mypy}
python=${PYTHON:-python3}

for tool in "$ruff" "$mypy" "$python"; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        echo "lint.sh: '$tool' not found (pip install ruff mypy, or set RUFF / MYPY / PYTHON)" >&2
        exit 2
    fi
done

cfg=(--config "$here/ruff.toml")
if [[ "${1:-}" != "--check" ]]; then
    "$ruff" check --fix --quiet "${cfg[@]}" "$here" || true
    "$ruff" format --quiet "${cfg[@]}" "$here"
fi
"$ruff" check "${cfg[@]}" "$here"
"$ruff" format --check "${cfg[@]}" "$here"
(cd "$root" && MYPYPATH="$root" "$mypy" --config-file "$here/mypy.ini" \
    --python-executable "$(command -v "$python")" -p examples.fhe_ssm_from_primitives)
