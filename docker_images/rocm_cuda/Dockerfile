FROM mixa3607/rocm-gfx906:7.2.1-complete

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        cmake \
        ninja-build \
        git \
        libcurl4-openssl-dev \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-9 \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb

ENV PATH=/usr/local/cuda-12.9/bin:/opt/rocm/bin:$PATH
