# Stage 1: Build the frontend
FROM node:14 AS frontend

WORKDIR /app/frontend
COPY react frontend/package.json react frontend/package-lock.json ./
RUN npm install
COPY react frontend/ ./
RUN npm run build

# Stage 2: Set up the backend
FROM python:3.9 AS backend

WORKDIR /app/backend
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ ./

# Stage 3: Combine both
FROM nginx:alpine

COPY --from=frontend /app/frontend/build /usr/share/nginx/html
COPY --from=backend /app/backend /app/backend

# Expose the port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
