"""
streamlit_app/app.py
Day49：金融 RAG 评测管理界面 —— 完整版（含美化、异常捕获、帮助说明、评测历史、红队扫描）
"""
import streamlit as st
import yaml
import pandas as pd
import time
import statistics
import json
import sys
import os
import re
import tempfile
import subprocess
from pathlib import Path

# 确保能导入项目根目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.eval_components import evaluate_planning, evaluate_retrieval, evaluate_generation

# ── 页面配置 ──────────────────────────────────────────
st.config.set_option("theme.base", "light")
st.set_page_config(
    page_title="金融 RAG 评测中心",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 金融 RAG 评测中心")
st.markdown("上传评测数据集，选择评测框架，一键运行并查看报告。")

# ── 自定义 CSS：卡片颜色 ──────────────────────────────
st.markdown("""
<style>
div[data-testid="stMetric"] {
    border-radius: 10px;
    padding: 10px;
}
.metric-pass div[data-testid="stMetricValue"] {
    color: #27ae60 !important;
}
.metric-fail div[data-testid="stMetricValue"] {
    color: #e74c3c !important;
}
</style>
""", unsafe_allow_html=True)

# ── 初始化 session_state ─────────────────────────────
for key, default in [
    ("uploaded_file", None),
    ("framework", "DeepEval"),
    ("results", None),
    ("progress", 0.0),
    ("error_message", ""),
    ("promptfoo_api_key", ""),
    ("run_clicked", False),
    ("promptfoo_html", None),
    ("redteam_html", None),          # 新增：红队报告
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 历史记录初始化
if "history" not in st.session_state:
    st.session_state.history = []

# ── 使用说明 ──────────────────────────────────────────
with st.expander("📖 使用说明"):
    st.markdown("""
    1. **上传评测数据集**：点击左侧「上传评测数据集」按钮，选择一个符合格式的 YAML 文件。
       - 文件必须包含 `categories` 字段，每个类别下有 `entries`，每个条目至少包含 `query` 和 `expected_answer`。
    2. **选择评测框架**：
       - **DeepEval**：对上传的用例执行组件级评估（规划、检索、生成），展示指标卡片和柱状图。
       - **Promptfoo**：使用 GPT-4o 进行多模型对比测试，生成 HTML 报告。需提供 OpenAI API Key（也可通过环境变量 `OPENAI_API_KEY` 读取）。
    3. **点击「开始评测」**：系统将根据选择的框架执行评估，进度条会实时更新。
    4. **查看结果**：评测完成后，主区域会显示指标卡片、图表和下载按钮。对于 Promptfoo，可下载 HTML 报告或在页面内预览。
    """)

# ── 合并所有侧边栏内容 ────────────────────────────────
with st.sidebar:
    st.header("🔧 评测配置")

    uploaded_file = st.file_uploader(
        "上传评测数据集 (.yaml)",
        type=["yaml", "yml"],
        help="请上传符合格式的评测数据集 YAML 文件",
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.results = None
            st.session_state.promptfoo_html = None
            st.session_state.redteam_html = None
            st.session_state.error_message = ""
            st.session_state.run_clicked = False

    framework = st.selectbox(
        "选择评测框架",
        options=["DeepEval", "Promptfoo"],
        index=0 if st.session_state.framework == "DeepEval" else 1,
        disabled=st.session_state.uploaded_file is None,
    )
    st.session_state.framework = framework

    if framework == "Promptfoo":
        st.session_state.promptfoo_api_key = st.text_input(
            "OpenAI API Key (用于 Promptfoo 评测)",
            type="password",
            value=st.session_state.get("promptfoo_api_key", ""),
            help="也可通过环境变量 OPENAI_API_KEY 自动读取",
        )
        st.session_state.run_redteam = st.checkbox(
            "启用红队安全扫描",
            value=False,
            help="额外运行 Promptfoo 红队扫描，消耗更多API配额"
        )

    run_button = st.button(
        "🚀 开始评测",
        disabled=st.session_state.uploaded_file is None,
        use_container_width=True,
    )
    if run_button:
        st.session_state.run_clicked = True
        st.session_state.error_message = ""
        st.session_state.results = None
        st.session_state.promptfoo_html = None
        st.session_state.redteam_html = None

    st.divider()
    st.caption("💡 评测框架说明：")
    st.caption("- **DeepEval**：组件级评估（规划/检索/生成）")
    st.caption("- **Promptfoo**：多模型对比与安全红队测试")

    # ── 红队报告下载 ──────────────────────────────────
    if st.session_state.get("redteam_html"):
        st.divider()
        st.subheader("🛡️ 红队扫描报告")
        st.download_button(
            "📥 下载红队报告",
            data=st.session_state.redteam_html,
            file_name="promptfoo_redteam_report.html",
            mime="text/html",
        )

    # ── 评测历史 ──────────────────────────────────────
    st.divider()
    st.subheader("📜 评测历史")
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            st.caption(f"{entry['timestamp']} | {entry['framework']} | 忠实度: {entry['faithfulness']:.2f}")
    else:
        st.caption("暂无评测记录")

# ── 主区域：评测集预览 ────────────────────────────────
if st.session_state.uploaded_file is not None:
    st.subheader("📋 评测集预览")
    try:
        st.session_state.uploaded_file.seek(0)
        yaml_content = yaml.safe_load(st.session_state.uploaded_file)
        test_cases = []
        if isinstance(yaml_content, list):
            test_cases = yaml_content
        elif isinstance(yaml_content, dict):
            for key in ["test_cases", "entries", "final_qa_dataset", "categories"]:
                if key in yaml_content:
                    data = yaml_content[key]
                    if isinstance(data, list):
                        if key == "categories":
                            for cat in data:
                                test_cases.extend(cat.get("entries", []))
                        else:
                            test_cases = data
                        break
        if not test_cases:
            st.error("❌ 评测集格式错误：未找到有效的测试用例，请检查 YAML 结构。")
        else:
            rows = []
            for i, case in enumerate(test_cases[:50]):
                query = case.get("query") or case.get("question", "")
                expected = case.get("expected_answer") or case.get("answer", "")
                category = case.get("category", "")
                rows.append({
                    "序号": i + 1,
                    "类别": category,
                    "问题": query[:60] + ("..." if len(query) > 60 else ""),
                    "预期答案": expected[:60] + ("..." if len(expected) > 60 else ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=300)
            st.caption(f"共 {len(test_cases)} 条测试用例，预览前 {min(len(test_cases), 50)} 条。")
    except yaml.YAMLError as e:
        st.session_state.error_message = f"YAML 解析错误：{e}"
        st.error(st.session_state.error_message)
    except Exception as e:
        st.session_state.error_message = f"文件处理失败：{e}"
        st.error(st.session_state.error_message)
else:
    st.info("👆 请在左侧上传一个评测数据集文件（.yaml），开始使用。")

# ── 评测执行函数 ──────────────────────────────────────
def run_deepeval_eval(uploaded_file):
    """执行 DeepEval 组件评估并返回报告"""
    try:
        uploaded_file.seek(0)
        content = yaml.safe_load(uploaded_file)
        cases = []
        if isinstance(content, list):
            cases = content
        elif isinstance(content, dict):
            for key in ["test_cases", "entries", "final_qa_dataset", "categories"]:
                if key in content:
                    data = content[key]
                    if isinstance(data, list):
                        if key == "categories":
                            for cat in data:
                                cases.extend(cat.get("entries", []))
                        else:
                            cases = data
                        break
        if not cases:
            return None
        import random
        if len(cases) > 20:
            cases = random.sample(cases, 20)  # 只评测 20 条
        total = len(cases)
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        detailed = []
        metrics_collector = {
            "planning": {"tool_accuracy": [], "arg_reasonableness": []},
            "retrieval": {"contextual_recall": [], "contextual_precision": []},
            "generation": {"faithfulness": [], "answer_relevancy": []},
        }

        for i, case in enumerate(cases):
            query = case.get("query") or case.get("question", "")
            if not query:
                continue

            # 规划
            plan_res = evaluate_planning(query=query, expected_tool="financial_qa",
                                         expected_args={"query": query})
            # 检索关键信息提取
            key_info = None
            retrieval_mark = case.get("retrieval", {})
            if isinstance(retrieval_mark, dict):
                key_info = retrieval_mark.get("key_info")
            if not key_info:
                expected = case.get("expected_answer") or case.get("answer", "")
                numbers = re.findall(r'\d+\.?\d*%?', expected)
                if numbers:
                    key_info = numbers
            ret_res = evaluate_retrieval(query=query, key_info=key_info)
            gen_res = evaluate_generation(query=query, expected_answer=case.get("expected_answer"))

            detail = {
                "id": case.get("id", f"case_{i}"),
                "query": query,
                "planning": plan_res,
                "retrieval": ret_res,
                "generation": gen_res,
            }
            detailed.append(detail)
            for k, v in plan_res.items():
                metrics_collector["planning"][k].append(v)
            for k, v in ret_res.items():
                metrics_collector["retrieval"][k].append(v)
            for k, v in gen_res.items():
                metrics_collector["generation"][k].append(v)

            progress_bar.progress((i + 1) / total)
            status_text.text(f"已评估 {i+1}/{total} 个用例...")
            time.sleep(0.05)

        progress_bar.empty()
        status_text.empty()

        def summarize(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
            }

        components_summary = {}
        for comp, met_dict in metrics_collector.items():
            comp_sum = {}
            for met, vals in met_dict.items():
                comp_sum[met] = summarize(vals)
            components_summary[comp] = comp_sum

        report = {
            "total_test_cases": total,
            "components_summary": components_summary,
            "detailed_results": detailed,
        }
        return report
    except Exception as e:
        st.error(f"DeepEval 评估失败：{e}")
        return None

def run_promptfoo_eval(uploaded_file, api_key=None, run_redteam=False):
    """执行 Promptfoo 评测（可选红队扫描），返回 HTML 字节数据。"""
    config_path = None
    output_html_path = None
    redteam_output_path = None

    try:
        # ── 1. 解析用例（同前） ──────────────────────
        uploaded_file.seek(0)
        content = yaml.safe_load(uploaded_file)
        cases = []
        if isinstance(content, list):
            cases = content
        elif isinstance(content, dict):
            for key in ["test_cases", "entries", "final_qa_dataset", "categories"]:
                if key in content:
                    data = content[key]
                    if isinstance(data, list):
                        if key == "categories":
                            for cat in data:
                                cases.extend(cat.get("entries", []))
                        else:
                            cases = data
                        break
        if not cases:
            raise ValueError("未找到有效测试用例")

        tests = []
        for case in cases:
            query = case.get("query") or case.get("question", "")
            if not query:
                continue
            test_entry = {"vars": {"query": query}, "assert": []}
            keywords = []
            if "expected_keywords" in case:
                kw = case["expected_keywords"]
                if isinstance(kw, str):
                    keywords.append(kw)
                elif isinstance(kw, list):
                    keywords.extend(kw)
            if not keywords:
                expected = case.get("expected_answer") or case.get("answer", "")
                numbers = re.findall(r'\d+\.?\d*%?', expected)
                keywords.extend(numbers)
            if keywords:
                test_entry["assert"].append({"type": "contains", "value": keywords[0]})
            tests.append(test_entry)

        # ── 2. 生成主评测配置 ────────────────────────
        config = {
            "description": "金融 RAG 评测",
            "providers": [{"id": "openai:gpt-4o", "config": {"apiKey": api_key or "${OPENAI_API_KEY}"}}],
            "prompts": [{"id": "direct", "template": "你是一个金融法规专家。请回答以下问题：\n\n问题：{{query}}"}],
            "tests": tests,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            config_path = f.name

        # ── 3. 准备输出文件 ──────────────────────────
        output_html_fd, output_html_path = tempfile.mkstemp(suffix=".html")
        os.close(output_html_fd)

        env = os.environ.copy()
        if api_key:
            env["OPENAI_API_KEY"] = api_key

        # ── 4. 运行主评测 ────────────────────────────
        cmd = f"npx promptfoo@latest eval -c {config_path} --output {output_html_path} --no-progress-bar"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"主评测失败: {result.stderr}")

        with open(output_html_path, "rb") as f:
            html_bytes = f.read()

        # ── 5. 可选红队扫描 ──────────────────────────
        if run_redteam:
            redteam_config = {
                "plugins": [
                    "injection", "jailbreak", "harmful", "rbac", "hijacking", "excessive-agency"
                ],
                "target": {"id": "openai:gpt-4o", "config": {"apiKey": api_key or "${OPENAI_API_KEY}"}},
                "numConcurrentRequests": 3,
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix="_redteam.yaml", delete=False) as f:
                yaml.dump(redteam_config, f, allow_unicode=True)
                redteam_config_path = f.name

            redteam_output_fd, redteam_output_path = tempfile.mkstemp(suffix="_redteam.html")
            os.close(redteam_output_fd)

            redteam_cmd = f"npx promptfoo@latest redteam run -c {redteam_config_path} --output {redteam_output_path} --no-progress-bar"
            redteam_result = subprocess.run(redteam_cmd, shell=True, capture_output=True, text=True, env=env, timeout=600)
            if redteam_result.returncode != 0:
                st.warning(f"红队扫描部分失败: {redteam_result.stderr}")
                # 红队失败不阻断主报告
            else:
                # 将红队报告作为额外下载保存到 session_state（或附加到 HTML）
                with open(redteam_output_path, "rb") as f:
                    st.session_state.redteam_html = f.read()
                st.success("红队扫描完成，可在侧边栏下载报告。")

        return html_bytes

    except subprocess.TimeoutExpired:
        st.error("评测超时，请减少用例数或检查网络。")
        return None
    except Exception as e:
        st.error(f"Promptfoo 评测失败：{e}")
        return None
    finally:
        # 清理所有临时文件
        for path in [config_path, output_html_path, redteam_config_path if run_redteam else None, redteam_output_path if run_redteam else None]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

# ── 运行按钮触发评测 ─────────────────────────────────
if st.session_state.run_clicked and st.session_state.uploaded_file is not None:
    if st.session_state.results is None and st.session_state.promptfoo_html is None:
        if st.session_state.framework == "DeepEval":
            with st.spinner("正在运行 DeepEval 组件评估..."):
                results = run_deepeval_eval(st.session_state.uploaded_file)
            if results:
                st.session_state.results = results
        else:
            with st.spinner("正在运行 Promptfoo 评测（可能需要几分钟）..."):
                api_key = st.session_state.promptfoo_api_key or os.getenv("OPENAI_API_KEY")
                html_bytes = run_promptfoo_eval(st.session_state.uploaded_file, api_key, run_redteam=st.session_state.get("run_redteam", False))
                if html_bytes:
                    st.session_state.promptfoo_html = html_bytes
        st.session_state.run_clicked = False
        st.rerun()

# ── 结果展示 ──────────────────────────────────────────
if st.session_state.results:
    st.subheader("📈 组件评估结果")
    report = st.session_state.results
    comps = report["components_summary"]

    # 定义阈值
    thresholds = {
        "planning": {"tool_accuracy": 0.9, "arg_reasonableness": 0.8},
        "retrieval": {"contextual_recall": 0.7, "contextual_precision": 0.7},
        "generation": {"faithfulness": 0.8, "answer_relevancy": 0.8},
    }

    # 指标卡片（带颜色）
    cols = st.columns(3)
    for idx, (comp, metrics) in enumerate(comps.items()):
        with cols[idx]:
            st.markdown(f"**{comp.upper()}**")
            for met, stats in metrics.items():
                mean_val = stats["mean"]
                th = thresholds.get(comp, {}).get(met, 0.5)
                pass_fail = "✅" if mean_val >= th else "❌"
                col_class = "metric-pass" if mean_val >= th else "metric-fail"
                st.markdown(f'<div class="{col_class}">{pass_fail} {met}: {mean_val:.2f}</div>', unsafe_allow_html=True)

    # Plotly 柱状图
    import plotly.graph_objects as go
    categories = []
    values = []
    colors = []
    for comp, metrics in comps.items():
        for met, stats in metrics.items():
            categories.append(f"{comp}.{met}")
            values.append(stats["mean"])
            th = thresholds.get(comp, {}).get(met, 0.5)
            colors.append("#27ae60" if stats["mean"] >= th else "#e74c3c")
    fig = go.Figure([go.Bar(x=categories, y=values, marker_color=colors,
                            text=[f"{v:.2f}" for v in values], textposition='auto')])
    fig.update_layout(title="各组件指标均值", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # 下载按钮
    json_str = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button("📥 下载 JSON 报告", json_str, "component_eval_report.json", "application/json")

    # 将结果存入历史
    summary = {
        "framework": "DeepEval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": report["total_test_cases"],
        "faithfulness": comps["generation"]["faithfulness"]["mean"],
        "answer_relevancy": comps["generation"]["answer_relevancy"]["mean"],
    }
    if not st.session_state.history or st.session_state.history[-1]["timestamp"] != summary["timestamp"]:
        st.session_state.history.append(summary)
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]

if st.session_state.promptfoo_html:
    st.subheader("📈 Promptfoo 评测结果")
    st.download_button("📥 下载 HTML 报告", data=st.session_state.promptfoo_html,
                       file_name="promptfoo_report.html", mime="text/html")
    with st.expander("🔍 在线预览"):
        st.components.v1.html(st.session_state.promptfoo_html, height=800, scrolling=True)
    # Promptfoo 结果也记录到历史
    summary = {
        "framework": "Promptfoo",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases": "N/A",
        "faithfulness": "N/A",
        "answer_relevancy": "N/A",
    }
    if not st.session_state.history or st.session_state.history[-1]["timestamp"] != summary["timestamp"]:
        st.session_state.history.append(summary)
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]