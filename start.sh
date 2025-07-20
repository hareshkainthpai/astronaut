#!/bin/bash

# Start Redis server in background
redis-server --daemonize yes --bind 0.0.0.0 --port 6379

# Wait for Redis to start
sleep 2

# Run Django migrations
python3 manage.py makemigrations
python3 manage.py migrate

# Create static directories if they don't exist
mkdir -p static staticfiles media

# Only run collectstatic in production or if COLLECT_STATIC is set
if [ "$DJANGO_ENV" = "production" ] || [ "$COLLECT_STATIC" = "true" ]; then
    echo "Collecting static files..."
    python3 manage.py collectstatic --noinput
else
    echo "Skipping collectstatic (development mode)"
fi

# Start Django server
python3 manage.py runserver 0.0.0.0:8000