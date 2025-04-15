# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend

# Copy package files and install dependencies first (leverages cache)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --no-cache # Add --no-cache or equivalent for cleaner installs if possible

# Copy remaining frontend code and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Django backend dependencies & collect static files
FROM python:3.11-slim-bullseye AS backend-build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend

# Install OS dependencies needed for building python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache

# Install Python dependencies (copy only requirements first for caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code AFTER installing dependencies
COPY backend/ ./
COPY endpoints/ ./ # Adjust if 'endpoints' is located elsewhere relative to Dockerfile

# Collect static files (ensure STATIC_ROOT is correctly set in settings.py)
# Example assumes STATIC_ROOT = BASE_DIR / "staticfiles_collected"
RUN python manage.py collectstatic --noinput --clear

# Stage 3: Final image - Python Alpine base + Nginx + Supervisor
FROM python:3.11-alpine AS final
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/backend

# Install runtime OS dependencies for Nginx, Supervisor, and Python packages (e.g., psycopg2)
RUN apk update && \
    apk add --no-cache \
    nginx \
    supervisor \
    libpq # Runtime dependency for psycopg2 (if used)

# --- Option A: Copy pre-installed packages (Faster build, potentially fragile glibc -> musl) ---
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend-build /usr/local/bin /usr/local/bin
# --- Option B: Reinstall packages (Slower build, safer for glibc -> musl) ---
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt
# --- Choose Option A OR Option B based on testing ---

# Copy application code
COPY --from=backend-build /app/backend /app/backend
COPY --from=backend-build /app/endpoints /app/endpoints # If needed at runtime

# Copy collected static files from backend build stage to a location Nginx will serve
COPY --from=backend-build /app/backend/staticfiles_collected /app/static_collected # Adjust source based on STATIC_ROOT

# Copy built frontend files to Nginx default serve location
COPY --from=frontend-build /app/frontend/build /usr/share/nginx/html

# Copy configuration files
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create directories for logs if needed by supervisord.conf
RUN mkdir -p /var/log/supervisor

# --- Recommended: Add non-root user for security ---
# RUN addgroup -S appgroup && adduser -S appuser -G appgroup
# RUN chown -R appuser:appgroup /app /var/log/supervisor /usr/share/nginx/html # Add other owned paths
# USER appuser
# --- Ensure Nginx/Supervisor/Gunicorn configs run as 'appuser' if enabled ---

EXPOSE 80

# Start Supervisor to manage Nginx and the backend process (e.g., gunicorn/uvicorn)
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]