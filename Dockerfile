# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
# Assuming your build output goes to /app/frontend/build
RUN npm run build

# Stage 2: Setup Django backend
FROM python:3.11-slim AS backend-build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY backend/ ./backend
COPY endpoints/ ./endpoints
# Ensure STATIC_ROOT is set in your Django settings.py, e.g., STATIC_ROOT = BASE_DIR / 'staticfiles'
# Assume STATIC_ROOT resolves to /app/backend/staticfiles
RUN python manage.py collectstatic --noinput --clear

# Stage 3: Final image with Nginx, Supervisor, Python runtime, and application
FROM nginx:alpine

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python runtime and supervisor
RUN apk update && \
    apk add --no-cache \
    python3 \
    py3-pip \
    supervisor

# Copy installed Python packages from backend-build stage
# Adjust python version if necessary (matches the version in backend-build)
COPY --from=backend-build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy Python executables (like gunicorn, django-admin)
COPY --from=backend-build /usr/local/bin /usr/local/bin

# Copy backend application code
COPY --from=backend-build /app/backend /app/backend

# Copy collected Django static files (adjust path if your STATIC_ROOT is different)
COPY --from=backend-build /app/backend/staticfiles /app/backend/staticfiles

# Copy built frontend from stage 1 (adjust build output dir if needed)
COPY --from=frontend-build /app/frontend/build /usr/share/nginx/html

# Copy Nginx configuration (you MUST create this file)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy Supervisor configuration (you MUST create this file)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set the working directory (optional, can be set in supervisord.conf too)
WORKDIR /app/backend

# Expose only the Nginx port
EXPOSE 80

# Run supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]