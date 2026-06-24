#!/usr/bin/env bash
# capture_k8s.sh — screenshot wrapper for Kubernetes + Slurm commands
# Usage: ./scripts/capture_k8s.sh <output_name> <command...>
#
# Runs the command on the local shell (kubectl, docker, etc.),
# renders the output as a Dracula-themed terminal screenshot.

set -euo pipefail

OUTPUT_NAME="${1:-screenshot}"
shift
CMD="$*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SCREENSHOT_DIR="$REPO_DIR/screenshots"
mkdir -p "$SCREENSHOT_DIR"

WIN_TEMP="C:\\Windows\\Temp"
WIN_TEMP_LINUX="/mnt/c/Windows/Temp"
HTML_FILE="k8s_capture_$$.html"
PNG_FILE="${OUTPUT_NAME}.png"

# Execute the command and capture ANSI output
INNER_HTML=$(eval "$CMD" 2>&1 | docker run --rm -i ghcr.io/nicowillis/ansi2html:latest --partial --scheme=dracula 2>/dev/null || eval "$CMD" 2>&1 | python3 -c "
import sys, html
lines = sys.stdin.read()
print(html.escape(lines).replace('\n','<br>'))
")

LINE_COUNT=$(echo "$INNER_HTML" | wc -l)
WIN_HEIGHT=$(( LINE_COUNT * 22 + 120 ))
[[ $WIN_HEIGHT -lt 200 ]] && WIN_HEIGHT=200

CMD_ESCAPED=$(echo "$CMD" | sed 's/</\&lt;/g; s/>/\&gt;/g')

cat > "$WIN_TEMP_LINUX/$HTML_FILE" << HTMLEOF
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { background: #282a36; display: inline-block; width: 900px; }
  .terminal-window { background: #282a36; width: 900px; }
  .terminal-body {
    padding: 12px 16px 16px 16px;
    color: #f8f8f2;
    font-family: 'Cascadia Code','Consolas','Courier New',monospace;
    font-size: 13px; line-height: 1.55; white-space: pre;
  }
  .prompt-line { margin-bottom: 4px; }
  .u  { color: #50fa7b; } .at { color: #f8f8f2; }
  .h  { color: #8be9fd; } .p  { color: #f8f8f2; } .c  { color: #f8f8f2; }
  pre { background: transparent !important; }
</style></head><body>
<div class="terminal-window"><div class="terminal-body">
<div class="prompt-line"><span class="u">naman</span><span class="at">@</span><span class="h">k8s-hpc</span><span class="p">:~\$ </span><span class="c">${CMD_ESCAPED}</span></div>
${INNER_HTML}
</div></div></body></html>
HTMLEOF

"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
  --headless=new --disable-gpu \
  --screenshot="${WIN_TEMP}\\${PNG_FILE}" \
  --window-size="900,${WIN_HEIGHT}" \
  --force-device-scale-factor=2 \
  --hide-scrollbars \
  "file:///${WIN_TEMP//\\/\/}/${HTML_FILE}" 2>/dev/null

cp "$WIN_TEMP_LINUX/$PNG_FILE" "$SCREENSHOT_DIR/$PNG_FILE"
rm -f "$WIN_TEMP_LINUX/$HTML_FILE" "$WIN_TEMP_LINUX/$PNG_FILE"
echo "Saved: $SCREENSHOT_DIR/$PNG_FILE"
