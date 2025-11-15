#!/bin/bash
set -euo pipefail

CMD=(
  python apply_prompts.py \
    --source-url https://uhglepenoqhfwhxgebru.supabase.co \
    --source-key sb_secret_msr2RLr9XBVmLfv1HfbwWQ_AyQQOTMj \
    --dest-table interesting_messages \
    --engine ollama \
    --prompts PROMPT_1 PROMPT_2 \
    --batch-size 200 \
    --insert-batch-size 25 \
    --start-date 2024-01-01 \
    --end-date 2025-03-30
)

while true; do
  "${CMD[@]}"
  status=$?
  if [ $status -eq 0 ]; then
    echo "Job completed successfully"
    exit 0
  fi
  echo "Command failed with status $status. Restarting in 5 seconds..."
  sleep 600
done
