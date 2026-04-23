#!/usr/bin/env bash
# Auto-resume wrapper for SafetyDemographicsInteractionModel.
# Biogeme sometimes silently exits after the "low draws" warning. We detect
# that by counting how many group summary .tex files exist in the checkpoint
# directory, and re-run with --checkpoint until all 17 groups have one.

set -u

ROOT=/data/koichi/cycling_safety_perception
MODEL_PICKLE="$ROOT/reports/models/mxl_choice_20260420_151037/final_full_model.pickle"
CHECKPOINT_DIR="$ROOT/reports/models/interaction/safety_demographics_20260420_182818"
LOG="$ROOT/logs/demographics.log"
MAX_ATTEMPTS=20
TOTAL_GROUPS=17

cd "$ROOT"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  done_count=$(find "$CHECKPOINT_DIR" -name "demographics_interaction_model_*.tex" 2>/dev/null | wc -l)
  echo "[wrapper] attempt $((attempt+1))/$MAX_ATTEMPTS — $done_count / $TOTAL_GROUPS groups complete" | tee -a "$LOG"

  if [ "$done_count" -ge "$TOTAL_GROUPS" ]; then
    echo "[wrapper] all $TOTAL_GROUPS groups done; exiting" | tee -a "$LOG"
    exit 0
  fi

  PYTHONPATH="$ROOT/cycling_safety_svi/modeling" \
    "$ROOT/.venv/bin/python" -u \
    "$ROOT/cycling_safety_svi/modeling/safety_demographics_interaction_model.py" \
    --model_path "$MODEL_PICKLE" \
    --checkpoint "$CHECKPOINT_DIR" \
    2>&1 | tee -a "$LOG"

  attempt=$((attempt+1))
done

echo "[wrapper] hit MAX_ATTEMPTS=$MAX_ATTEMPTS, giving up" | tee -a "$LOG"
exit 1
