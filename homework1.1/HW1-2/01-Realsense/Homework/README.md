### Note: This guide is currently tailored for macOS. Updates for other operating systems are welcome.

### Setup ubuntu vm with colima

### Install Colima via Homebrew

brew install colima

### Start Colima with x86_64 emulation

colima start --arch x86_64 --cpu 4 --memory 8

### Check docker version

docker version

### Using Ubuntu 22.04 (LTS)

docker pull ubuntu:22.04

### Start a container interactively

docker run -it --name realsense-dev ubuntu:22.04 bash

### install dependecies

apt update && apt install -y \
 python3 python3-pip python3-dev \
 build-essential cmake git libusb-1.0-0-dev pkg-config \
 wget curl

apt update
apt install -y python3 python3-pip python3-venv python3-dev build-essential cmake git libusb-1.0-0-dev pkg-config

### Create the virtual enviroment

python3 -m venv pyrealsense-env
source pyrealsense-env/bin/activate

### Update pip

pip install --upgrade pip

### install realsense2

pip install pyrealsense2

### Persist your environment

### Open a new terminal

### Commit the container as a new image so you don’t have to rebuild:

### docker commit realsense-dev realsense-image

#### Step 1: Install VS Code Extensions

1.  Open VS Code
2.  Install dev-conteiners extension

docker run -it --name realsense-dev -v "<your_localproject_path>:/workspace" realsense-image:latest bash
