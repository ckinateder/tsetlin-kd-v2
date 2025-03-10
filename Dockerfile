FROM python:3.10.16-slim-bullseye

# dependencies 
RUN apt update && apt install -y build-essential git libgl1 libglib2.0-0 curl wget fontconfig fonts-cmu

# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install -y git-lfs
# env
ENV OMP_NUM_THREADS=18

# install msttcorefonts
RUN apt-get install -y  cabextract xfonts-utils
RUN apt --fix-broken install
RUN wget http://ftp.us.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.8.1_all.deb
RUN dpkg -i ttf-mscorefonts-installer_3.8.1_all.deb

# pip requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip ipykernel
RUN pip install --no-cache-dir -r /app/requirements.txt

# set workdir
WORKDIR /app