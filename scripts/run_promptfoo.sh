#!/bin/bash
# scripts/run_promptfoo.sh
# 一键运行 Promptfoo 多模型对比评估 + 红队安全扫描 + 生成报告
set -e

echo "🚀 启动 Promptfoo 评测流水线..."

# 检查必要的环境变量
: "${OPENAI_API_KEY:?请设置 OPENAI_API_KEY}"
: "${ANTHROPIC_API_KEY:?请设置 ANTHROPIC_API_KEY}"
: "${DEEPSEEK_API_KEY:?请设置 DEEPSEEK_API_KEY}"

# 1. 多模型 RAG 对比评估
echo "📊 执行多模型对比评估..."
npx promptfoo@latest eval -c promptfooconfig.yaml --output reports/promptfoo/rag_comparison.html
echo "✅ 多模型对比报告已生成: reports/promptfoo/rag_comparison.html"

# 2. 红队安全扫描（针对 GPT-4o 或设置的目标模型）
echo "🛡️ 执行红队安全扫描..."
npx promptfoo@latest redteam run \
  --plugins injection,jailbreak,harmful,rbac,hijacking,excessive-agency \
  --target openai:gpt-4o \
  --num-concurrent 3 \
  --output reports/promptfoo/redteam_report.html
echo "✅ 红队安全扫描报告已生成: reports/promptfoo/redteam_report.html"

echo "🎉 所有评测完成，报告位于 reports/promptfoo/ 目录。"