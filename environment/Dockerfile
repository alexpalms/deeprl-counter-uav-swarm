FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for tkinter and matplotlib
RUN apt-get update && apt-get install -y \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from the current directory to the container
COPY counter_drone_swarm/environment/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to environment.py
ENTRYPOINT ["python", "environment.py"]
