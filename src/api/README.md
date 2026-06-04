# 金融 RAG Agent API

基于 FastAPI 构建的异步 API 服务，将金融法规 RAG 问答和评测能力封装为 RESTful 端点，支持高并发与流式响应。

## 启动方式

### 本地开发

```bash
# 安装依赖
pip install -r api/requirements.txt

# 启动服务（自动重载）
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 构建镜像
docker build -t finance-rag-api:latest -f api/Dockerfile .

# 运行容器
# docker run -d --name finance-api \
#   -p 8000:8000 \
#   -e DASHSCOPE_API_KEY=your_key \
#   -e REDIS_URL=redis://host.docker.internal:6379 \
#   -v $(pwd)/chroma_db:/app/chroma_db \
#   finance-rag-api:latest


docker run -d --name finance-api \
  -p 8000:8000 \
  -e DASHSCOPE_API_KEY=sk-991aa8d5210f42fab50ce7f59dfca11a \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  -e VECTOR_STORE_BACKEND=pgvector \
  -e PGVECTOR_CONNECTION_STRING="postgresql+psycopg://pgvector:pgvector@host.docker.internal:5433/ai_rag" \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  -v /mnt/d/work/pythonai/finance_agent/models:/app/models \
  finance-rag-api:latest

  启动后访问 http://localhost:8000/docs 查看交互式 Swagger 文档。