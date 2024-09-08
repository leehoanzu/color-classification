# python:3.11-bookworm
FROM dtcooper/raspberrypi-os:latest

WORKDIR /home

# Copy files to the container
COPY . .

# Update package list and install packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install pip==22.3.1 --break-system-packages && \ 
    pip install opencv-python-headless && \
    pip install pyyaml && \
    pip install torch torchvision && \
    pip install onnx onnxruntime


# Command to run the application (uncomment if needed)
CMD ["python3", "./utils/main.py"]

