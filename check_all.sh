#!/usr/bin/env bash
# check_all.sh -- 一条命令跑完全部验证。
#
# 设计原则（2026-10-17 升级后总结）：
#   1. 不依赖网络。任何需要联网的检查都必须能从缓存降级。
#   2. 每项独立报告；任何一项非零即整体非零。
#   3. 总耗时可控：默认跳过最慢的两项，用 --full 打开。
#
# 用法:
#   bash demo/check_all.sh            # 快速（跳过 verify_paper_numbers 与 validate）
#   bash demo/check_all.sh --full     # 全跑（约 5-10 分钟）
set -uo pipefail
cd "$(dirname "$0")"

FULL=0
[ "${1:-}" = "--full" ] && FULL=1
fail=0
run() {
  local name="$1"; shift
  printf '  %-26s ' "$name"
  if out=$("$@" 2>&1); then
    echo "$(echo "$out" | grep -oE 'ALL TESTS PASSED|通过 [0-9]+|可追溯: *[0-9]+' | head -1)"
  else
    echo "❌ 失败"; echo "$out" | tail -5 | sed 's/^/      /'; fail=1
  fi
}

echo "═══ 第一层：组件自检 ═══"
run "run_demo selfcheck" python3 run_demo.py selfcheck

echo "═══ 第二层：数字与 CSV 一致 ═══"
run "verify_numbers" python3 verify_numbers.py

echo "═══ 方向 B 的数字追溯 ═══"
run "verify_B" python3 verify_B.py --quiet

echo "═══ 跨文件一致性 ═══"
run "audit_consistency" python3 audit_consistency.py --quiet
run "crosscheck_copies" python3 crosscheck_copies.py --quiet

if [ "$FULL" = "1" ]; then
  echo "═══ 第三层：论文级追溯（慢）═══"
  run "verify_paper_numbers" python3 verify_paper_numbers.py
  echo "═══ 第四层：数据本身对不对（慢）═══"
  run "validate" python3 validate.py
else
  echo "（跳过 verify_paper_numbers 与 validate；加 --full 打开）"
fi

echo
if [ "$fail" = "0" ]; then echo "═══ 全部通过 ═══"; else echo "═══ 有失败项 ═══"; fi
exit "$fail"
