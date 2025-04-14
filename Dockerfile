# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY react\ frontend/package.json react\ frontend/package-lock.json ./
RUN npm install
COPY react\ frontend/ ./
RUN npm run build


FROM python:3.11-slim AS backend-build

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app/backend

COPY react\ frontend/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY backend/ ./backend
COPY backend/chatbot ./backend/chatbot
COPY endpoints/ ./endpoints

# Collect static files (if any)
RUN python backend/manage.py collectstatic --noinput || true

# Stage 3: Final image with Nginx and Gunicorn
FROM nginx:alpine

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

# Copy backend app
COPY --from=backend-build /app/backend /app/backend

# Copy Nginx config (optional, can be added later)
# COPY nginx.conf /etc/nginx/nginx.conf

# Expose ports
EXPOSE 80
EXPOSE 8000

# Start Gunicorn and Nginx
CMD sh -c "gunicorn backend.wsgi:application --bind 0.0.0.0:8000 & nginx -g 'daemon off;'"
