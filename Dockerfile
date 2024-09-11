# Use a base image with Python 3.9
FROM python:3.12

# Set the working directory
WORKDIR /app

# Install system dependencies including CMake and build tools
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and set a longer timeout for pip install
RUN pip install --upgrade pip
RUN pip config set global.timeout 120

# Install dlib before other requirements
RUN pip install cmake
RUN pip install --upgrade pip setuptools wheel
RUN pip install dlib

RUN pip install praat-parselmouth

# Copy the requirements file and install other dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "main_server:app"]
