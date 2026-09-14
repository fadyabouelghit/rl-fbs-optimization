#!/usr/bin/env bash
# Wait for the equivalence workflow (5 agents) to finish, confirm the machine is
# quiet, then run the full-length faithful run alone. Nothing overlaps.
# NOTE: grep -c / pgrep -c print "0" AND exit 1 when there are no matches, so
# they must NOT be combined with '|| echo 0' -- that yields "0\n0".
J="/Users/fadya/.claude/projects/-Users-fadya-Documents-MATLAB-GA-github-genetic-algorithm-optimization/0181f6e5-aae6-4df5-af8c-33d9637265d4/subagents/workflows/wf_50b9d3c5-7ad/journal.jsonl"
count_results() { grep -c '"type":"result"' "$J" 2>/dev/null | head -1; }
count_python()  { pgrep -f 'rl-fbs-pyqd/venv/bin/python' 2>/dev/null | wc -l | tr -d ' '; }
echo "[queue] waiting for equivalence workflow (5 agents) ..."
while [ "$(count_results)" -lt 5 ]; do sleep 20; done
echo "[queue] workflow done $(date +%H:%M:%S); waiting for machine to go quiet ..."
QUIET=0
while [ "$QUIET" -lt 4 ]; do
  if [ "$(count_python)" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  sleep 15
done
echo "[queue] launching full faithful run $(date +%H:%M:%S)"
exec ./run_full_faithful.sh
