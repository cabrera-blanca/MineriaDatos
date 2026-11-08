FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencias primero: esta capa se reutiliza mientras requirements-api.txt no cambie
COPY requirements-api.txt .
# Caché de descargas (no queda en la imagen) y timeouts amplios: con redes lentas un corte no obliga a empezar de cero
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries=10 -r requirements-api.txt

# Código y modelo (el modelo se obtiene con `dvc pull` o `python -m src.inference.export_model` antes del build)
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Usuario sin privilegios
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
