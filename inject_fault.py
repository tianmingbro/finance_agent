# inject_fault.py
from eval_components import evaluate_generation, _get_eval_llm
import json

# 备份原始函数
original_generation = evaluate_generation

# 替换为会降低忠实度的版本（修改 Prompt 使其忽略上下文）
def fault_generation(query, expected_answer=None):
    # 先调用原始函数获取正常结果（确保上下文存在）
    # 但修改内部的 LLM 调用，让答案加入虚构内容
    # 这里为简单，我们直接修改 evaluate_generation 中使用的 prompt
    # 更优雅的方式是 mock 掉 LLM 的 invoke 返回值
    import os
    from langchain_openai import ChatOpenAI
    from src.retriever.tools_mcp import search_finance_docs
    import asyncio
    
    # 获取上下文
    result_json = asyncio.run(search_finance_docs(query, top_k=4))
    docs_data = json.loads(result_json)
    retrieval_context = [doc["content"] for doc in docs_data.get("documents", [])]
    if not retrieval_context:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
    
    # 构造一个会“故意编造”的 LLM
    llm = ChatOpenAI(model="qwen-plus", temperature=0.9,  # 提高温度鼓励编造
                     openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
                     openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
    context_str = "\n".join([f"文档{i+1}: {c}" for i, c in enumerate(retrieval_context)])
    # 故意指示模型添加虚构信息
    prompt = f"你是一个金融法规专家。根据以下信息回答问题，可以适当补充推测的内容：\n\n{context_str}\n\n问题：{query}\n答案："
    answer = llm.invoke(prompt).content
    
    # 仍然使用相同的指标计算
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    test_case_faith = LLMTestCase(input=query, actual_output=answer, retrieval_context=retrieval_context)
    faith_metric = FaithfulnessMetric(model=_get_eval_llm())
    faith_metric.measure(test_case_faith)
    test_case_rel = LLMTestCase(input=query, actual_output=answer)
    rel_metric = AnswerRelevancyMetric(model=_get_eval_llm())
    rel_metric.measure(test_case_rel)
    return {"faithfulness": faith_metric.score, "answer_relevancy": rel_metric.score}

# 替换函数
import eval_components
eval_components.evaluate_generation = fault_generation

# 重新生成报告
from pathlib import Path
import yaml
from eval_components import generate_component_report

with open('data/component_eval_data.yaml', 'r', encoding='utf-8') as f:
    test_data = yaml.safe_load(f)['test_cases']

fault_report = generate_component_report(test_data)
print("注入错误后的生成组件指标：")
gen_metrics = fault_report['components_summary']['generation']['metrics']
print(f"  忠实度均值: {gen_metrics['faithfulness']['mean']:.3f}")
print(f"  答案相关性均值: {gen_metrics['answer_relevancy']['mean']:.3f}")