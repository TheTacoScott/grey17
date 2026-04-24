FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV BLENDER_VERSION=5.1.0
ENV BLENDER_PATH=/usr/local/blender
ENV PATH="${BLENDER_PATH}:${PATH}"

RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
    python3-pip \
    ffmpeg \
    libchromaprint-tools \
    libxrender1 \
    libxxf86vm1 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libsm6 \
    libgl1-mesa-glx \
    libgles2-mesa \
    libglib2.0-0 \
    libfontconfig1 \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    ImageHash \
    Pillow \
    pyyaml \
    jsonschema

RUN wget -q \
    https://download.blender.org/release/Blender5.1/blender-5.1.0-linux-x64.tar.xz \
    && tar -xf blender-5.1.0-linux-x64.tar.xz \
    && mv blender-5.1.0-linux-x64 ${BLENDER_PATH} \
    && rm blender-5.1.0-linux-x64.tar.xz

COPY scripts/ /scripts/

WORKDIR /work
