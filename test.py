import requests
try:
    r = requests.get("http://localhost:8000/health", timeout=10)
    print(r.text)
except Exception as e:
    print(f"连接失败: {e}")