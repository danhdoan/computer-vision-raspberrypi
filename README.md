# Computer Vision on Raspberry Pi

by Danh Doan


## Introduction
This is a series about developing common Computer Vision projects on Raspberry Pi board. Some of them requires the support of Movidius Neural Compute Stick to boost the performance. 
OpenVINO toolkit is mainly the development tool that helps optimize the hardware and models to work well with Raspi

The main purpose of this work is to help developers from all levels to gain insights and resources to working with Raspberry Pi for Computer Vision projects.
Every sample project is refactored and organized so that it is easily understandable and approachable.

If you have any project ideas and issues with those projects, feel free to comment.
It will help improve and enrich the contents. Thanks in advance with my sincere.

**Other Computer Vision demos:** [[link]](https://www.youtube.com/watch?v=Suprnm2EiEE&list=PL9gpyuNNKEhJSAg8RxTrNj046GQJ1K9Q1)

## Updates
**2019, Nov 27:**
  * Add 012-tflite-object-detection

**2019, Nov 15:**
  * Add `openvino-models`: [[link]](https://drive.google.com/drive/folders/11G98FS2-klB4qGiz4YzEZfdnd0G2XNrL)

**2019, Nov 05:**
  * Add 008-pi-emotion-recognition
  * Update installation guide to support IECore

**2019, Oct 23:**
  * Add 004-pi-head-pose-estimation

**2019, Oct 16:**
  * Add 006-pi-face-verification

**2019, Oct 15:**
  * Add 003-pi-face-alignment

**2019, Oct 12:**
  * Add 005-pi-object-detection
  * Add 002-pi-facial-landmark-detection
  * Add 001-pi-face-detection
  * Add 000-show-pi-camera

## Sample Projects
* Test with Pi camera module: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/000-show-pi-camera)
	* Play around with builtin Pi camera module
* Face Detection with High Accuracy: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/001-pi-face-detection)
	* Develop an accurate and robust Face detector with pretrained SSD model trained from WIDER dataset
* Facial Keypoint Detection: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/002-pi-facial-landmark-detection) [[demo]](https://www.youtube.com/watch?v=En_nsyF8kJM) [[demo]](https://www.youtube.com/watch?v=WzvgrhrDC1s)
	* Develop a simple Facial keypoints localization that detect 5 main keypoints of human faces (center eyes, nose tip, and mouth corner
* Face Alignment: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/003-pi-face-alignment)
	* Based on 5 keypoints, align human faces, to support other problem e.g. Face Identification, Face Verification, ... 
* Headpose Estimation: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/004-pi-head-pose-estimation) [[demo]](https://www.youtube.com/watch?v=kN-QrA3h4oo)
	* Estimate Human head pose in Tait-Bryan angles (yaw, pitсh or roll)

* Human Detection: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/005-pi-object-detection) [[demo]](https://www.youtube.com/watch?v=Suprnm2EiEE)
	* Develop an Object detector especially with SSD model

	**Notice:** human is just an example of objects, any object detection model can be converted to work with this sample project

* Face Verification: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/006-pi-face-identification)

  Verify Face Identity by Face embeddings

* Emotion Recognition: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/008-pi-emotion-recognition) [[demo]](https://www.youtube.com/watch?v=RXCuG3I1Mkw)

  Recognize Emotional states of Human faces

* Car and License Plate Detection: [ongoing]

* TFLite Object Detection: [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/tree/master/012-tflite-object-detection) [[demo]](https://www.youtube.com/watch?v=ncDyjjNTd5w)



## Installation

Follow `install.md` instructions [[link]](https://github.com/danhdoan/computer-vision-raspberrypi/blob/master/install.md) to install essential packages and modules for working with Raspberry Pi. Installing OpenVINO as a toolkit to the development.

## Usage
1. Clone this repository:

	`cd ~`

	`mkdir workspace && cd workspace`

	`git clone https://github.com/danhdoan/computer-vision-raspberrypi`

2. Download OpenVINO pretrained-model

	`mkdir openvino-models tflite-models`

	You can notice a soft symbol link in any projects that maps to this directory. If you want to store it elsewhere, beware of re-map this symbol link.
	To download a model, just go to the official OpenVINO site from Intel:

	https://download.01.org/opencv/2019/open_model_zoo/R1/

	In this project, models from R1 sub-dir are used. R3 is currently the latest, it also works well with sample code.
	Just download the FP16 models, they can be applied to Raspberry Pi. In my code, I usually add a postfix `-fp16` to clarify this issue. Thus, download a pretrained model and change its name correspondingly.
	
3. Run a sample project

	All sample projects have the same argument parser

		usage: <app-name>.py [-h] [--video VIDEO] [--name NAME] [--show]                              
	                              [--record] [--flip_hor] [--flip_ver]
		optional arguments:                                                                                    
		  -h, --help            show this help message and exit
		  --video VIDEO, -v VIDEO
		                        Video Streamming link or Path to video source
		  --name NAME, -n NAME  Name of video source
		  --show, -s            Whether to show the output visualization
		  --record, -r          Whether to save the output visualization
		  --flip_hor, -fh       horizontally flip video frame
		  --flip_ver, -fv       vertically flip video frame
		  
	It is implemented in `altusi/helper.py`, you can customize as you wanted. To see argument list of a sample, e.g. `object-detection`, just type:

		python3 app-object-detector.py -h

## References
* OpenVINO model zoo: https://download.01.org/opencv/2019/open_model_zoo/R3/
* Official Intel site:
	* [https://docs.openvinotoolkit.org/latest/_demos_README.html](https://docs.openvinotoolkit.org/latest/_demos_README.html)
	* https://docs.openvinotoolkit.org/latest/_models_intel_index.html
* Raspberry Pi and OpenVINO installation: https://www.hackster.io/news/getting-started-with-the-intel-neural-compute-stick-2-and-the-raspberry-pi-6904ccfe963

