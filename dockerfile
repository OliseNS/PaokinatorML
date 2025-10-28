FROM python:3.11-slim as builder

# Install uv (the fast dependency manager)
RUN pip install uv

# Create a virtual environment in a standard location
RUN uv venv /opt/venv

# Copy ONLY the locked requirements file
COPY requirements.txt .

# Install dependencies into the venv using uv
RUN uv pip install --no-cache-dir --python /opt/venv/bin/python -r requirements.txt

# This stage builds the final, slim container
FROM python:3.11-slim as final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY . .

CMD sh -c "gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:${PORT:-8000}"