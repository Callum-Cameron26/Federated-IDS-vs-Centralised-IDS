FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY app /workspace/app

ENV DATA_DIR=/data
ENV RUNS_DIR=/runs
ENV PYTHONPATH=/workspace

CMD ["python", "-m", "app.main"]
