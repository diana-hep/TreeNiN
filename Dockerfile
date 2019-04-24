#
# Dockerfile for the TreeNiN
# To build the docker image run:
# docker build --tag=treenin:1.0.0 .


FROM phusion/baseimage:0.9.19
MAINTAINER Sebastian Macaluso

#Download base image ubuntu 16.04
#FROM ubuntu:16.04

#FROM rootproject/root-ubuntu16

USER root 

#Set time zone
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
apt-get -y install vim \
bc                           \
curl                         \
git                          \
wget                         \
python-pip                   \
python3-pip                   \
libx11-dev                   \
libxpm-dev                   \
libxft-dev                   \
libxext-dev                  \
libpng3                      \
libjpeg8                     \
gfortran                     \
libssl-dev                   \
libpcre3-dev                 \
libgl1-mesa-dev              \
libglew1.5-dev               \
libftgl-dev                  \
libmysqlclient-dev           \
libfftw3-dev                 \
libcfitsio3-dev              \
graphviz-dev                 \
libavahi-compat-libdnssd-dev \
libldap2-dev                 \
libxml2-dev  &&              \
apt-get clean && \
rm -rf /var/lib/apt/lists/*


# install all pip-able requirements
COPY scripts/requirements.txt /tmp/
COPY scripts/requirements3.txt /tmp/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install tables
RUN pip3 install --no-cache-dir -r /tmp/requirements3.txt


# INSTALL FASTJET
COPY scripts/install-fastjet.sh /tmp/
RUN chmod +x /tmp/install*

ENV NUMCORES=2
RUN /tmp/install-fastjet.sh $NUMCORES


WORKDIR /TreeNiN

COPY README.md /TreeNiN/README.md
COPY code /TreeNiN/code

#Go to the main directory
WORKDIR /TreeNiN/code/

# Then run:
# python dataWorkflow.py 0
# python MLWorkflow.py 0 9

