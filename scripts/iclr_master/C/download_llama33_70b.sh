#!/bin/bash
# Route Q needs the proposing model of the H_R build (meta-llama/Llama-3.3-70B-Instruct). It is not on this box
# and there is no HF token; ModelScope mirrors it publicly (LLM-Research/Llama-3.3-70B-Instruct).
# Every file is checked against the sha256 the ModelScope API lists.
set -u
DST=/mnt/pfs/devs/pn5wp/shishuqing/models/Llama-3.3-70B-Instruct
REPO=LLM-Research/Llama-3.3-70B-Instruct
mkdir -p $DST; cd $DST
curl -s "https://www.modelscope.cn/api/v1/models/$REPO/repo/files?Revision=master&Recursive=true" > files.json
/root/venvs/partnr/bin/python - <<'PY' > manifest.tsv
import json
d=json.load(open("files.json"))["Data"]["Files"]
for f in d:
    if f["Type"]=="blob" and not f["Path"].startswith("original/"):
        print(f"{f['Path']}\t{f['Sha256']}\t{f['Size']}")
PY
echo "files: $(wc -l < manifest.tsv)"
# fetch in parallel (one stream is ~6 MB/s here), then verify sequentially below
cut -f1,3 manifest.tsv | while IFS=$'\t' read -r path size; do
  [ -f "$path" ] && [ "$(stat -c %s "$path")" = "$size" ] || echo "$path"
done | xargs -P 10 -I{} curl -sSL -C - --retry 5 -o {} "https://www.modelscope.cn/models/$REPO/resolve/master/{}"
fail=0
while IFS=$'\t' read -r path sha size; do
  for try in 1 2 3 4 5; do
    if [ -f "$path" ] && [ "$(stat -c %s "$path")" = "$size" ]; then break; fi
    curl -sSL -C - --retry 5 -o "$path" "https://www.modelscope.cn/models/$REPO/resolve/master/$path"
  done
  got=$(sha256sum "$path" | cut -d' ' -f1)
  if [ -n "$sha" ] && [ "$got" != "$sha" ]; then echo "SHA MISMATCH $path"; fail=1; else echo "ok $path"; fi
done < manifest.tsv
echo "DONE fail=$fail"
