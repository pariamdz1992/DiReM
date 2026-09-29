#!/usr/bin/env bash
# Upload files into an existing Zenodo draft, with a checksum check for each file.
#
#   read -rs ZENODO_TOKEN && export ZENODO_TOKEN   # paste your token (it is not echoed)
#   bash zenodo_upload.sh <draft_id> file1 [file2 ...]
#
# <draft_id> is the number in the draft's address, https://zenodo.org/uploads/<draft_id>.
# The token comes from Zenodo > Applications > Personal access tokens, with the scopes
# deposit:write and deposit:actions. The draft stays a draft: publish it on the website.
set -euo pipefail
: "${ZENODO_TOKEN:?set ZENODO_TOKEN first}"
ID=${1:?draft id}; shift
AUTH="Authorization: Bearer $ZENODO_TOKEN"
BUCKET=$(curl -sf -H "$AUTH" "https://zenodo.org/api/deposit/depositions/$ID" \
         | python3 -c 'import sys, json; print(json.load(sys.stdin)["links"]["bucket"])')
for f in "$@"; do
  echo "uploading $f ($(du -h "$f" | cut -f1))"
  resp=$(curl -f --progress-bar -H "$AUTH" --upload-file "$f" "$BUCKET/$(basename "$f")")
  remote=$(printf '%s' "$resp" | python3 -c 'import sys, json; print(json.load(sys.stdin)["checksum"].split(":")[-1])')
  local_md5=$(md5sum "$f" | cut -d' ' -f1)
  if [ "$remote" = "$local_md5" ]; then echo "  ok (md5 $local_md5)"; else echo "  CHECKSUM MISMATCH: $f"; exit 1; fi
done
