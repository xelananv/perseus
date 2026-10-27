# Upstream files this branch changes

The fhe-ssm port is built on upstream's `speedrun` branch. It adds files under `python/perseus/examples/fhe_ssm/` and
`tools/fhe_ssm/`, and it may change upstream files as fixes the port needs. Every changed upstream file is listed below,
one row per file, its path in backticks in the first column. `tools/fhe_ssm/check_additive.sh` accepts a changed
upstream file only when it is listed here and fails on any other; `tools/fhe_ssm/sync_upstream.sh` runs that check after
every rebase and names the listed files that upstream itself changed since the last sync, so each fix is re-checked.

| path | change | reason | upstream status |
|---|---|---|---|
