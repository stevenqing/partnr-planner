#!/bin/bash
# Status of part C runs on the remote box. Run there: bash status.sh
P=/mnt/pfs/devs/pn5wp/shishuqing
O=$P/partnr-isambard-C/outputs/iclr_master_C
S=$P/partnr-planner/scripts/iclr_master/C
date
echo "--- processes"
ps -eo pid,etime,cmd | grep '[l]ane.sh\|[r]un_cell.sh' | cut -c1-150
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2
echo "--- cells"
for d in $O/*/; do n=98; case $(basename $d) in idcheck*|smoke*) n=5;; esac
  /root/venvs/partnr/bin/python $S/cell_status.py ${d%/} $n; done
echo "--- episodes used: $(ls $O/*/hydra/results/*/stats/*.json 2>/dev/null | wc -l) / 1700"
echo "--- queue.log tail"; tail -5 $O/queue.log 2>/dev/null
