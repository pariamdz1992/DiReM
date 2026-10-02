#!/usr/bin/env bash
# Upload files into an existing Zenodo draft, with a checksum check for each file.
#
#   read -rs ZENODO_TOKEN && export ZENODO_TOKEN   # paste your token (it is not echoed)
#   bash zenodo_upload.sh <draft_id> file1 [file2 ...]
#
# <draft_id> is the number in the draft's address, https://zenodo.org/uploads/<draft_id>.
# The token comes from Zenodo > Applications > Personal access tokens, with the scopes
# deposit:write and deposit:actions. The draft stays a draft: publish it on the website.
#
# Safe to re-run: a file already in the draft with the same MD5 is skipped. Run it inside
# tmux (or screen) so a dropped connection does not stop a long upload.
set -euo pipefail
: "${ZENODO_TOKEN:?set ZENODO_TOKEN first}"
ID=${1:?draft id}; shift
AUTH="Authorization: Bearer $ZENODO_TOKEN"
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

# 1. open the draft: its upload bucket and the files it already holds
code=$(curl -sS -o "$TMP" -w '%{http_code}' -H "$AUTH" "https://zenodo.org/api/deposit/depositions/$ID") || {
  echo "cannot reach zenodo.org (no internet on this machine? on a cluster, use a login node)"; exit 1; }
if [ "$code" != 200 ]; then
  echo "cannot open draft $ID (HTTP $code):"; cat "$TMP"; echo
  echo "401/403: token wrong or missing the deposit:write scope. 404: wrong draft number."
  exit 1
fi
BUCKET=$(python3 -c 'import sys, json; print(json.load(open(sys.argv[1]))["links"]["bucket"])' "$TMP")
EXISTING=$(python3 -c '
import sys, json
for f in json.load(open(sys.argv[1])).get("files", []):
    print(f.get("checksum", "").split(":")[-1], f.get("filename", ""))' "$TMP")
echo "draft $ID open; it already holds $(printf '%s' "$EXISTING" | grep -c . || true) file(s)"

# 2. upload each file that is not there yet
for f in "$@"; do
  name=$(basename "$f")
  echo "== $name ($(du -h "$f" | cut -f1)): computing MD5"
  local_md5=$(md5sum "$f" | cut -d' ' -f1)
  if printf '%s\n' "$EXISTING" | grep -qx "$local_md5 $name"; then
    echo "  already uploaded, skipped"; continue
  fi
  code=$(curl -sS --progress-bar --retry 3 -o "$TMP" -w '%{http_code}' -H "$AUTH" \
              --upload-file "$f" "$BUCKET/$name") || { echo "  upload interrupted: re-run the script"; exit 1; }
  if [ "$code" != 200 ] && [ "$code" != 201 ]; then
    echo "  upload failed (HTTP $code):"; cat "$TMP"; echo; exit 1
  fi
  remote=$(python3 -c 'import sys, json; print(json.load(open(sys.argv[1]))["checksum"].split(":")[-1])' "$TMP")
  if [ "$remote" = "$local_md5" ]; then echo "  ok (md5 $local_md5)"; else echo "  CHECKSUM MISMATCH: $name"; exit 1; fi
done
echo "all files are in draft $ID; check them at https://zenodo.org/uploads/$ID, then publish there"
