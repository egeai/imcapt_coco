# Use an NVIDIA CUDA base image compatible with the CUDA 12.2
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


# Install dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3 \
    python3-pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


WORKDIR /app


# Copy the requirements file into the container at /code
COPY ./requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu122

# Copy the current directory contents into the container at /app
COPY . .

ENV PYTHONPATH=/app

# CMD [ "python3", "./src/data/make_dataset.py" ]
# ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
