FROM python:3.12-slim

# Copy the uv binary from the official Astral uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Forces logs to print immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync with caching for faster builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
