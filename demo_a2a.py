#!/usr/bin/env python3
"""
demo_a2a.py
启动三个 A2A Agent 服务，模拟用户交互：
  1. 提问“资本充足率是多少？”
  2. 对回答进行评测
"""
import subprocess
import time
import sys
import requests
import json
import os
import signal

# 配置
RAG_PORT = 8101
EVAL_PORT = 8102
ORCH_PORT = 8100
BASE_URL = f"http://localhost:{ORCH_PORT}"
RAG_SCRIPT = "src/agent/a2a/rag_agent.py"
EVAL_SCRIPT = "src/agent/a2a/eval_agent.py"
ORCH_SCRIPT = "src/agent/a2a/orchestrator_agent.py"

processes = []

def start_agent(script, port, name):
    """在后台启动一个 Agent 服务"""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen(
        [sys.executable, script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(p)
    print(f"启动 {name} (PID {p.pid}) ...")
    # 等待服务就绪
    for _ in range(20):
        try:
            requests.get(f"http://localhost:{port}/.well-known/agent-card", timeout=1)
            print(f"{name} 已就绪 (端口 {port})")
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"{name} 启动超时")

def stop_all():
    for p in processes:
        p.terminate()
        p.wait()
    print("所有 Agent 已停止")

def make_task_request(query, session_id="demo"):
    """构造任务请求"""
    task = {
        "id": f"task-{int(time.time()*1000)}",
        "session_id": session_id,
        "messages": [
            {"role": "user", "parts": [{"text": query}]}
        ],
        "status": "created",
        "artifacts": []
    }
    return {"task": task}

def run_demo():
    try:
        # 启动 Agent 服务
        start_agent(RAG_SCRIPT, RAG_PORT, "RAG Agent")
        start_agent(EVAL_SCRIPT, EVAL_PORT, "Eval Agent")
        start_agent(ORCH_SCRIPT, ORCH_PORT, "Orchestrator")
        print("\n===== 多 Agent 协作演示开始 =====\n")

        session_id = "demo-session"
        # 1. 金融问答
        question = "资本充足率是多少？"
        print(f"用户: {question}")
        resp = requests.post(f"{BASE_URL}/a2a/task", json=make_task_request(question, session_id))
        result = resp.json()
        if result["status"] == "completed":
            answer = result["artifacts"][0].get("answer", "")
            print(f"助手: {answer}")
        else:
            print(f"调用失败: {result}")

        # 2. 评测（使用刚获得的答案）
        # 从 answer 中提取简要答案文本进行评测，这里直接假设答案为固定示例（或从步骤1获取）
        answer_to_eval = answer if 'answer' in locals() else "核心一级资本充足率不得低于5%"
        eval_query = f"评测 {question}|{answer_to_eval}"
        print(f"\n用户: 评测一下这个回答")
        resp = requests.post(f"{BASE_URL}/a2a/task", json=make_task_request(eval_query, session_id))
        result = resp.json()
        if result["status"] == "completed":
            eval_data = result["artifacts"][0]
            print(f"评测结果: 忠实度 {eval_data.get('faithfulness')}, 答案相关性 {eval_data.get('answer_relevancy')}")
        else:
            print(f"评测失败: {result}")

        print("\n===== 演示结束 =====")
    finally:
        stop_all()

if __name__ == "__main__":
    run_demo()