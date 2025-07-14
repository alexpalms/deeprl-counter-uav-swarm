FROM python:3.11-slim

WORKDIR /app

# Copy all files from the current directory to the container
COPY counter_drone_swarm/agent/ .

# Install requirements if the file exists
RUN if [ -f requirements ]; then pip install --no-cache-dir -r requirements; fi

# Set the entry point to distance_based_policy.py
ENTRYPOINT ["python", "distance_based_policy.py"]
