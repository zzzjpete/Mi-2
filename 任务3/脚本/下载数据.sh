#!/usr/bin/env bash
# 顺序下载 oa_comm baseline 包 PMC001..PMC010（约 81.5GB），支持断点续传。
# 已完整下载（本地大小 == 远端 Content-Length）的包自动跳过。
set -u
BASE="https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml"
DEST="E:/rag/data/pubmed"
DATE="2026-06-18"
cd "$DEST" || exit 1

for p in 001 002 003 004 005 006 007 008 009 010; do
  fn="oa_comm_xml.PMC${p}xxxxxx.baseline.${DATE}.tar.gz"
  url="$BASE/$fn"
  remote=$(curl -sI --max-time 60 "$url" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
  local=0; [ -f "$fn" ] && local=$(stat -c%s "$fn" 2>/dev/null || echo 0)
  if [ -n "$remote" ] && [ "$local" = "$remote" ]; then
    echo "[skip] $fn already complete ($remote bytes)"
    continue
  fi
  echo "[get ] $fn  (have $local / $remote bytes)  $(date '+%H:%M:%S')"
  # -C - 断点续传；--retry 应对网络抖动
  curl -sS -C - --retry 10 --retry-delay 5 --max-time 0 -o "$fn" "$url"
  local2=$(stat -c%s "$fn" 2>/dev/null || echo 0)
  if [ -n "$remote" ] && [ "$local2" = "$remote" ]; then
    echo "[done] $fn  ($local2 bytes)  $(date '+%H:%M:%S')"
  else
    echo "[warn] $fn incomplete: $local2 / $remote  $(date '+%H:%M:%S')"
  fi
done
echo "ALL DOWNLOADS FINISHED $(date '+%H:%M:%S')"
