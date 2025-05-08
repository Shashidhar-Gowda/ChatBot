#!/bin/bash

echo "🔍 Checking for running MongoDB containers..."

# List all running containers with mongo image
mongo_container=$(docker ps --filter "ancestor=mongo" --format "{{.Names}}")

if [ -z "$mongo_container" ]; then
  echo "❌ No MongoDB container is currently running."
  exit 1
else
  echo "✅ MongoDB container is running. Name: $mongo_container"
fi

echo "🌐 Checking if MongoDB is reachable at localhost:27017..."

# Try to connect to MongoDB port
if nc -z localhost 27017; then
  echo "✅ MongoDB is accessible at localhost:27017"
else
  echo "❌ MongoDB is NOT accessible on localhost:27017"
fi

