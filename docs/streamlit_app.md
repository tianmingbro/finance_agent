# 金融 RAG 评测管理界面

基于 Streamlit 构建的可视化评测管理工具，支持上传评测数据集、选择 DeepEval 或 Promptfoo 评测框架、一键运行并查看评测报告。

## 启动方式

```bash
# 安装依赖
pip install streamlit pyyaml pandas plotly

# 启动应用
streamlit run streamlit_app/app.py

启动后浏览器访问 http://localhost:8501。
功能列表

    文件上传：支持 .yaml 格式的评测数据集，自动解析并预览表格。

    评测框架选择：

        DeepEval：组件级评估（规划、检索、生成），展示指标卡片、柱状图和下载链接。

        Promptfoo：多模型对比测试，生成 HTML 报告，支持在线预览和下载。

    进度反馈：评测过程中显示进度条和状态文字。

    结果可视化：使用 Plotly 生成彩色柱状图，指标卡片根据阈值动态着色（通过绿色、未通过红色）。

    报告导出：支持 JSON 详细报告和 CSV 数据导出。

    使用说明：页面内置展开式帮助文档。

    技术细节

    状态管理：利用 st.session_state 持久化文件、框架、结果和进度。

    评测引擎：

        DeepEval 调用 eval_components.py 中的 evaluate_planning、evaluate_retrieval、evaluate_generation。

        Promptfoo 通过动态生成临时 promptfooconfig.yaml，使用 subprocess 调用 npx promptfoo eval 并捕获 HTML 报告。

    异常处理：全局 try/except 确保任何步骤失败时显示清晰错误提示，不崩溃。

    自定义样式：通过 CSS 为指标卡片添加颜色标识。
    
文件结构
text

streamlit_app/
  └── app.py          # Streamlit 主应用
eval_components.py    # DeepEval 组件评估函数
tools_mcp.py          # 检索工具（被 eval_components 依赖）
