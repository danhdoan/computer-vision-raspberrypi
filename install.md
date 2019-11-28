# Raspberry Pi Environment Setup Guide
by Danh Doan


## Setup Raspbian OS
1. Download Raspbian image from official site [[link]](https://www.raspberrypi.org/downloads/raspbian/)

  Currently use Raspbian Stretch with Desktop - Release date: 2019-08-04

2. Write image to SD card using balenaEtcher [[link]](https://www.balena.io/etcher/)

3. Insert SD card to Raspberry Pi and power it on

  After booting, configure Timezone and Language, then update system

4. Enable Peripherals e.g. SSH, Picamera, Audio or Communication protocols: I2C, SPI

	`sudo raspi-config`

	Select `Interfacing Options` and `Advanced Options` to enable

5. Install basic packages

	`sudo apt clean && sudo apt autoremove`

	`sudo apt update && sudo apt upgrade`

	`sudo apt install git cmake vim-gtk gedit`

## Setup OpenVINO toolkit
Follow the installation guide in this blog [[link]](https://blog.hackster.io/getting-started-with-the-intel-neural-compute-stick-2-and-the-raspberry-pi-6904ccfe963)

1. Download and Extract OpenVINO toolkit

	`cd ~`

	`mkdir openvino`

	`cd openvino`

	`wget https://download.01.org/opencv/2019/openvinotoolkit/R3/l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz`

	`tar -xvf l_openvino_toolkit_runtime_raspbian_p_2019.3.334.tgz`
  
  `mv l_openvino_toolkit_runtime_raspbian_p_2019.3.334/* .`

2. Modify Installation Dir in setup script

	`sed -i "s|<INSTALLDIR>|$(pwd)/|" bin/setupvars.sh`

3. Add to .bashrc

	`source /home/pi/openvino/bin/setupvars.sh`

	`source ~/.bashrc`


4. Update USB rule for Pi to recognize NCS

	`sudo usermod -a -G users "$(whoami)"`

	`sh ~/openvino/install_dependencies/install_NCS_udev_rules.sh`


## Install support packages for Computer Vision

`sudo apt install python3-picamera`
