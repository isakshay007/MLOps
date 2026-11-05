# syntax=docker/dockerfile:1
# Multi-stage build: builder installs the packages, test runs pytest, and the final image
# only gets the runtime packages and the code. The final stage copies a marker file from the
# test stage, so a failing test stops `docker build` before any runtime image is produced.
ARG PYTHON_VERSION=3.12

# ---- builder: install pinned runtime packages into /install ----
FROM python:${PYTHON_VERSION}-slim AS builder
WORKDIR /build
# Copy only the requirements first, so code changes don't re-run pip (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- base: slim runtime with the packages and a non-root user (no code yet) ----
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/output
COPY --from=builder /install /usr/local
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/output \
    && chown appuser:appuser /app/output
WORKDIR /app

# ---- test: run the test suite against the same code and packages as the runtime image ----
FROM base AS test
# pytest is installed before the code is copied, so code changes reuse this layer too
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt
COPY src/ ./src/
COPY tests/ ./tests/
USER appuser
RUN python -m pytest -q -p no:cacheprovider tests && touch /tmp/tests-passed

# ---- runtime: the image you run (no pytest, no tests) ----
FROM base AS runtime
COPY src/ ./src/
LABEL org.opencontainers.image.title="docker-lab1-diabetes" \
      org.opencontainers.image.description="Trains a Gradient Boosting regressor on the Diabetes dataset and predicts disease progression" \
      org.opencontainers.image.source="https://github.com/isakshay007/MLOps" \
      org.opencontainers.image.licenses="MIT"
COPY --from=test /tmp/tests-passed /tmp/tests-passed
USER appuser
# The trained model and metrics are written here; mount a host folder to keep them
VOLUME ["/app/output"]
ENTRYPOINT ["python", "src/main.py"]
CMD ["train"]
