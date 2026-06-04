FROM python:3.10-slim

WORKDIR /app


# 安装编译依赖（sentence-transformers 需要）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc g++ make && rm -rf /var/lib/apt/lists/*

# 配置 pip 国内镜像（关键！否则 PyTorch 可能下不动）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 先复制依赖文件（利用 Docker 缓存分层）
COPY src/api/requirements.txt .

# 安装依赖（这一层会被缓存，只要 requirements.txt 不变，以后都不会重新下载）
RUN pip install --no-cache-dir -r requirements.txt

# # 复制模型文件（从本地，不联网）
# COPY local_models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2

# # 设置模型路径
# ENV HF_HOME=/app/models
# ENV TRANSFORMERS_CACHE=/app/models

# 最后复制项目代码（代码经常变，放在最后避免使上面的缓存失效）
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]