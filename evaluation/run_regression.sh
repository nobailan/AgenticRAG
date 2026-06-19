#!/bin/bash
# AgenticRAG — 一键回归测试（本地运行）
# 用法: bash evaluation/run_regression.sh

set -e
cd "$(dirname "$0")/.."

echo "===== AgenticRAG 回归测试 ====="
echo ""

# 1. 运行回归测试
echo "[1/2] 运行回归测试..."
python evaluation/regression_test.py "$@"
EXIT_CODE=$?

# 2. 打印结果摘要
echo ""
echo "[2/2] 测试完成 (exit code=$EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 全部指标正常，无退化"
else
    echo "❌ 检测到退化，请检查上方输出"
fi

exit $EXIT_CODE
