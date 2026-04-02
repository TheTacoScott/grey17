FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV BLENDER_VERSION=2.91.0
ENV BLENDER_PATH=/usr/local/blender
ENV PATH="${BLENDER_PATH}:${PATH}"

RUN apt-get update && apt-get install -y \
    wget \
    xz-utils \
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

RUN wget -q \
    https://mirror.clarkson.edu/blender/release/Blender2.91/blender-2.91.0-linux64.tar.xz \
    && tar -xf blender-2.91.0-linux64.tar.xz \
    && mv blender-2.91.0-linux64 ${BLENDER_PATH} \
    && rm blender-2.91.0-linux64.tar.xz

COPY scripts/ /scripts/

WORKDIR /work
