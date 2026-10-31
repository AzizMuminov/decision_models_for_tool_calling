#!/usr/bin/env bash
# One-time sandbox setup for the coding-harness experiment (run as root).
#   - an unprivileged user `agentbox` runs the agent; its only writable area is /srv/agentbox/work
#   - the experiment repo (answers, world files, results) and the scratchpad (API key) are root-only
#   - opencode and a Python with pytest are installed read-only under /opt/agentbox
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p /srv/agentbox
id agentbox >/dev/null 2>&1 || useradd --create-home --home-dir /srv/agentbox/home --shell /bin/bash agentbox
chmod 700 "$REPO"

mkdir -p /opt/agentbox
rm -rf /opt/agentbox/opencode && mkdir -p /opt/agentbox/opencode
cp -r "$REPO/harness/node_modules" /opt/agentbox/opencode/
[ -x /opt/agentbox/py/bin/python ] || python3 -m venv /opt/agentbox/py
/opt/agentbox/py/bin/pip install -q pytest
chown -R root:root /opt/agentbox && chmod -R a+rX,go-w /opt/agentbox

mkdir -p /srv/agentbox/work
chown agentbox:agentbox /srv/agentbox/work && chmod 700 /srv/agentbox/work
echo "sandbox ready: user $(id agentbox)"
