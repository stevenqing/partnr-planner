#!/bin/bash
# Monitor jobs until completion
while true; do
  running=$(squeue -j 168394,168395,168396,168397,168398 -h 2>/dev/null | wc -l)
  if [ "$running" -eq 0 ]; then
    echo "All jobs completed at $(date)"
    break
  fi
  echo "$(date): $running jobs still running..."
  sleep 120
done
echo "=== Final Job Status ==="
sacct -j 168394,168395,168396,168397,168398 --format=JobID,JobName%20,State,Elapsed,ExitCode
