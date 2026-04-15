#!/usr/bin/env bash
# Run the full Ritrova scan + cluster + auto-assign pipeline and capture a
# benchmark report (timings, per-phase progress, before/after stats).
#
# Each phase logs to its own file; SUMMARY.txt aggregates the headlines.
# Failed phases do not abort the run — the report shows ok/FAILED per phase
# and you can rerun anything individually afterwards.
#
# Usage:
#   bash scripts/benchmark_scan.sh                       # default report dir
#   bash scripts/benchmark_scan.sh /path/to/report/dir   # custom report dir
#
# Output:
#   data/benchmarks/<timestamp>/SUMMARY.txt
#   data/benchmarks/<timestamp>/<phase>.log
#   data/benchmarks/<timestamp>/stats_{before,after}.txt

set -uo pipefail   # NOT -e: keep going past phase failures

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d-%H%M%S)"
REPORT_DIR="${1:-data/benchmarks/$TS}"
mkdir -p "$REPORT_DIR"

SUMMARY="$REPORT_DIR/SUMMARY.txt"
echo "Ritrova scan benchmark — $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee "$SUMMARY"
echo "Report dir: $REPORT_DIR"                                  | tee -a "$SUMMARY"

# Load .env so FACE_DB / PHOTOS_DIR are visible.
if [ -f .env ]; then
    set -a; source .env; set +a
fi
DB="${FACE_DB:-./data/faces.db}"
echo "DB:         $DB"                                          | tee -a "$SUMMARY"
echo "Photos:     ${PHOTOS_DIR:-<unset>}"                       | tee -a "$SUMMARY"

# Backup the DB up front.
BACKUP="$DB.pre-benchmark-$TS"
cp "$DB" "$BACKUP"
echo "Backup:     $BACKUP"                                      | tee -a "$SUMMARY"

# Run a phase: time it, capture stdout+stderr to its own log, append a one-line
# result to the summary that includes the last progress line of the log (which
# carries per-phase counters like processed=… faces=… errors=…).
run_phase() {
    local name="$1"; shift
    local log="$REPORT_DIR/${name}.log"
    local start; start=$(date +%s)
    echo                                                        | tee -a "$SUMMARY"
    echo "── $name — $(date '+%H:%M:%S') ──"                    | tee -a "$SUMMARY"
    if "$@" >"$log" 2>&1; then
        local result="ok"
    else
        local result="FAILED ($?)"
    fi
    local elapsed=$(( $(date +%s) - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    # Last non-empty line of the log, stripped of carriage returns from the
    # in-place progress prints (`\r [i/total] ...`).
    local last
    last="$(tr '\r' '\n' <"$log" | awk 'NF' | tail -1 | tail -c 240)"
    printf "%-22s %s  %3dm%02ds  | %s\n" \
        "$name" "$result" "$mins" "$secs" "$last" | tee -a "$SUMMARY"
}

echo                                                            | tee -a "$SUMMARY"
echo "═══ Stats BEFORE ═══"                                     | tee -a "$SUMMARY"
uv run ritrova stats | tee "$REPORT_DIR/stats_before.txt"        | tee -a "$SUMMARY"

# Standard pipeline. Order matches the README's recommended workflow.
run_phase "scan-photos"        uv run ritrova scan
run_phase "scan-videos"        uv run ritrova scan-videos
run_phase "scan-pets"          uv run ritrova scan-pets
run_phase "cluster"            uv run ritrova cluster
run_phase "auto-assign-people" uv run ritrova auto-assign
run_phase "auto-assign-pets"   uv run ritrova auto-assign --kind pet

echo                                                            | tee -a "$SUMMARY"
echo "═══ Stats AFTER ═══"                                      | tee -a "$SUMMARY"
uv run ritrova stats | tee "$REPORT_DIR/stats_after.txt"         | tee -a "$SUMMARY"

echo                                                            | tee -a "$SUMMARY"
echo "═══ Stats DIFF ═══"                                       | tee -a "$SUMMARY"
diff "$REPORT_DIR/stats_before.txt" "$REPORT_DIR/stats_after.txt" \
    | tee -a "$SUMMARY" || true

# Sanity: synthetic backfill scans from the migration should remain at 0 (real
# scans always create proper rows). A non-zero count would suggest something
# inserted findings outside the scanner pipeline.
echo                                                            | tee -a "$SUMMARY"
n_legacy=$(uv run ritrova scans list 2>/dev/null \
    | grep -c legacy_backfill || true)
echo "Synthetic legacy_backfill scans: $n_legacy (expect 0)"     | tee -a "$SUMMARY"

echo                                                            | tee -a "$SUMMARY"
echo "Done. Summary: $SUMMARY"                                  | tee -a "$SUMMARY"
