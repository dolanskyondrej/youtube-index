FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY ytindex ./ytindex
RUN pip install .
VOLUME ["/app/data"]
EXPOSE 8000
# ytindex.toml a data/ se připojí jako volume (viz docker-compose.yml)
CMD ["ytindex", "serve", "--host", "0.0.0.0", "--port", "8000"]
