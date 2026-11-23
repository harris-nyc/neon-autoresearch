# Only needed for AUTORESEARCH_SANDBOX=docker (if `unshare` isn't allowed on your box).
#   docker build -t neon-autoresearch:latest .
FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /opt/env
COPY pyproject.toml .
RUN uv sync --no-install-project && ln -s /opt/env/.venv/bin/python /usr/local/bin/python-venv
ENV PATH="/opt/env/.venv/bin:$PATH"
WORKDIR /work
