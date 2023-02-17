FROM python:3.9.16-slim

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /Propeller_inspection

COPY classes.txt ./
COPY propeller_fault_detection_v5.py ./
COPY config ./config
COPY models ./models
COPY requirements.txt ./requirements.txt

RUN mkdir -p ./input
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Minimize image size 
#RUN (apt-get autoremove -y; \
#     apt-get autoclean -y)
#ENV QT_X11_NO_MITSHM=1

#CMD ["bash"]
CMD [ "python", "propeller_fault_detection_v5.py" ]