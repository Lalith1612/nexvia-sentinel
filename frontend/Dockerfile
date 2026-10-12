# ==============================================================================
# Nexvia Sentinel — Frontend Production Dockerfile (Vite + Nginx)
# ==============================================================================
# Multi-stage build: Compile TypeScript/React application and serve via Nginx
# ==============================================================================

# Stage 1: Build static bundle
FROM node:20-alpine AS build

WORKDIR /app

# Copy dependency specifications and install cleanly
COPY package*.json ./
RUN npm ci

# Copy source code and build production assets
COPY . .
RUN npm run build

# Stage 2: Serve via hardened Nginx Alpine image
FROM nginx:alpine

# Remove default nginx HTML assets
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled frontend assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration with SPA routing and API reverse-proxying
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
