#!/usr/bin/env bash
# One-line status of everything the is_in_room line has in flight.
#   ssh aibox-root 'cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner && bash scripts/drivers/partnr_status.sh'
set -u
cd /mnt/pfs/devs/pn5wp/shishuqing/partnr-planner || exit 1
date '+%F %H:%M:%S'

cell () {                                   # $1 label  $2 dir
  local n=0
  [ -d "$2/results" ] && n=$(ls "$2"/results/*/stats 2>/dev/null | wc -l)
  printf '  %-22s %3s/369  %s\n' "$1" "$n" "$( [ -f "$2/CELL.json" ] && cat "$2/CELL.json" || echo 'running' )"
}

echo '-- band (decides whether +0.1093 is significant) --'
cell base_rep outputs/report/priv_iir1/base_rep
echo '-- model rerun --'
for c in m7b_base m7b_accepted m8b_base m8b_accepted; do cell "$c" "outputs/report/model_rerun/$c"; done
echo '-- comparisons written (read `complete` before any number) --'
ls outputs/report/priv_iir1/band_val_mini.json outputs/report/model_rerun/compare_*.json 2>/dev/null || echo '  none yet'
echo '-- endpoints --'
for u in 8061 8101; do printf '  %s ' $u; curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$u/v1/models; done
echo '-- drivers alive --'
ps -eo pid,etime,cmd | grep -E '[p]artnr_(model_rerun|band_val|model_cell|gate_cell)' | cut -c1-96
echo '-- load --'; uptime
