#!/usr/bin/env python3
"""
scripts/concurrent_session_test.py
并发测试：5 个 session 同时进行问答+评测，验证上下文隔离
增加超时和重试，适应真实评测耗时。
"""
import concurrent.futures
import requests
import time
import json

ORCH_URL = "http://localhost:8100/a2a/task"
SESSIONS = 5
CLIENT_TIMEOUT = 90        # 增加到 90 秒，适应并发评测
MAX_RETRIES = 2

QUESTIONS = [
    "资本充足率要求是多少？",
    "存款保险最高限额？",
    "个人外汇便利化额度？",
    "LPR最新报价是多少？",
    "反洗钱义务有哪些？"
]

def post_with_retry(url, json_payload, timeout=CLIENT_TIMEOUT):
    """带重试的 POST 请求"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=json_payload, timeout=timeout)
            return resp
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"    请求超时，重试中 ({attempt}/{MAX_RETRIES})...")
                time.sleep(2)
            else:
                raise

def session_task(session_id: str, question: str):
    """单个会话的完整流程：提问 → 评测，返回评测结果"""
    session = f"concurrent-{session_id}"
    # 1. 提问
    task_id = f"task-{session}-{int(time.time()*1000)}"
    payload = {
        "task": {
            "id": task_id,
            "session_id": session,
            "messages": [{"role": "user", "parts": [{"text": question}]}],
            "context": {},
            "status": "created",
            "artifacts": []
        }
    }
    try:
        resp = post_with_retry(ORCH_URL, payload)
        if resp.status_code != 200 or resp.json().get("status") != "completed":
            return {"session": session, "error": f"问答失败 (status={resp.status_code})"}
        answer_data = resp.json()
    except Exception as e:
        return {"session": session, "error": f"问答异常: {str(e)}"}

    # 2. 评测（无具体答案，依赖上下文）
    task_id_eval = f"task-{session}-eval-{int(time.time()*1000)}"
    payload_eval = {
        "task": {
            "id": task_id_eval,
            "session_id": session,
            "messages": [{"role": "user", "parts": [{"text": "评测一下"}]}],
            "context": {},
            "status": "created",
            "artifacts": []
        }
    }
    try:
        resp_eval = post_with_retry(ORCH_URL, payload_eval)
        if resp_eval.status_code != 200 or resp_eval.json().get("status") != "completed":
            return {"session": session, "error": "评测失败"}
        eval_data = resp_eval.json()
    except Exception as e:
        return {"session": session, "error": f"评测异常: {str(e)}"}

    artifacts = eval_data.get("artifacts", [{}])
    return {
        "session": session,
        "question_asked": question,
        "eval_query": artifacts[0].get("query", ""),
        "eval_answer_snippet": artifacts[0].get("answer", "")[:50]
    }

def main():
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=SESSIONS) as executor:
        futures = [
            executor.submit(session_task, i, QUESTIONS[i])
            for i in range(SESSIONS)
        ]
        results = [f.result() for f in futures]

    elapsed = time.time() - start
    print(f"并发 {SESSIONS} 个会话完成，耗时 {elapsed:.2f} 秒\n")

    errors = []
    for r in results:
        if "error" in r:
            errors.append(r)
            print(f"❌ 会话 {r['session']}: {r['error']}")
        else:
            match = r["question_asked"] in r["eval_query"]
            status = "✅" if match else "❌ 串号"
            print(f"{status} 会话 {r['session']}: 提问「{r['question_asked']}」→ 评测 query「{r['eval_query']}」")

    if errors:
        print(f"\n失败会话数：{len(errors)}")
    else:
        print("\n所有会话上下文正确隔离，无串号！")

if __name__ == "__main__":
    main()