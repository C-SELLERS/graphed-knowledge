FROM python:3.12-alpine AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-alpine

WORKDIR /app

COPY --from=builder /install /usr/local
COPY graphed_kb/ ./graphed_kb/
COPY mcp_server.py .
COPY mcp_admin.py .

EXPOSE 8000

CMD ["python", "mcp_server.py"]
