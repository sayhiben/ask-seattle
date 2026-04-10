FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ASK_SEATTLE_NO_WRITE=1
ENV ASK_SEATTLE_TORCH_DEVICE=cpu

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[reddit,transformer]"

CMD ["ask-seattle", "stream"]
