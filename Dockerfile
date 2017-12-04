FROM python:2
EXPOSE 8021 9098 9099 9100 9111 9090 10101 10102 10103
LABEL maintainer="vs1@andrew.cmu.edu"
LABEL description="A container that runs the gabriel server application \
 usage docker run gabriel \
if any parameters that has to be passed while starting the server can be passed in the following way: docker run gabriel <param 1>< param 2>"



#RUN apt-get install  clang cmake ccache
RUN apt-get update
RUN apt-get install -y default-jre
#RUN apt-get install -y python-setuptools python-dev build-essential python-pip
RUN apt-get install -y gcc pssh git musl-dev wget
RUN wget https://apertium.projectjj.com/apt/install-release.sh -O - | bash
RUN apt-get -y install locales build-essential automake subversion pkg-config gawk libtool apertium-all-dev
RUN pip install psutil
RUN pip install virtualenv
RUN virtualenv -p python2.7 ~/.env-2.7
RUN /bin/bash -c "source ~/.env-2.7/bin/activate"
RUN apt-get update && apt-get -y install tesseract-ocr
RUN pip install opencv-contrib-python
RUN git clone https://github.com/cmusatyalab/gabriel.git
RUN rm -r gabriel/client
RUN pip install -r gabriel/server/requirements.txt
RUN apertium-get en-es && cd apertium-en-es && make && make install
RUN pip install pillow pytesseract
RUN cd gabriel/server/bin/example-proxies && git clone https://github.com/vignesh380/mobilePervasiveDemp.git
#ENTRYPOINT /bin/sh
#ENTRYPOINT /gabriel/server/bin/gabriel-control > /gabriel/server/bin/logs
CMD python /gabriel/server/bin/gabriel-control
