FROM ubuntu:18.04

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
RUN bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install pytorch torchvision cpuonly -c pytorch
RUN mkdir /workspace/ && cd /workspace/ && git clone https://github.com/Omkar-Ranadive/Domain-Adaptation-CycleGAN.git && cd Domain-Adaptation-CycleGAN && pip install -r requirements.txt

WORKDIR /workspace