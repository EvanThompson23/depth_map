FROM ros:humble-ros-base as build
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python3-matplotlib
RUN apt-get -y install python3-opencv
RUN apt-get -y install python3-pil
RUN apt-get -y install ros-humble-cv-bridge
RUN mkdir /workspace
COPY ./src /workspace