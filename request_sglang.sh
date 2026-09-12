#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT="${1:-你好，请介绍一下自己}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "错误：找不到 Python ${PYTHON}" >&2
  exit 1
fi

export BASE_URL MODEL MAX_TOKENS PROMPT

exec "${PYTHON}" <<'PY'
import json
import os
import sys
import time
import urllib.error
import urllib.request

url = os.environ["BASE_URL"].rstrip("/") + "/v1/chat/completions"
payload = {
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(1, 6):
    try:
        with opener.open(request, timeout=120) as response:
            result = json.load(response)
        break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in (502, 503) and attempt < 5:
            print(f"服务暂未就绪（HTTP {exc.code}），2 秒后重试 {attempt}/5...", file=sys.stderr)
            time.sleep(2)
            continue
        print(f"请求失败：HTTP {exc.code}: {detail or exc.reason}", file=sys.stderr)
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print(f"请求失败：{exc}", file=sys.stderr)
        raise SystemExit(1)

try:
    print(result["choices"][0]["message"]["content"])
except (KeyError, IndexError, TypeError):
    print(json.dumps(result, ensure_ascii=False, indent=2))
PY


# curl http://127.0.0.1:43440/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Qwen/Qwen3-0.6B",
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant."},
#       {"role": "user", "content": "Tell me a short joke."}
#     ],
#     "max_tokens": 100
#   }'
