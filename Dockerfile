

FROM python:3.12-slim

ARG TORCH_VERSION=2.2.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md requirements-api.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}+cpu" \
    && python -m pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY configs ./configs

RUN python -m pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "ragops.app:app", "--host", "0.0.0.0", "--port", "8000"]
