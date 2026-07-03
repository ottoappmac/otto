FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        git \
        bash \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y vim
RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"
RUN sed -i 's/OSH_THEME=.*/OSH_THEME="cupcake"/' ~/.bashrc

# Add your application requirements
ADD requirements.txt /tmp/requirements.txt
ADD requirements-dev.txt /tmp/requirements-dev.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements-dev.txt

ENV PYTHONPATH="/workspace/src"
