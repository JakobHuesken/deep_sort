# Setup the detection site
Be sure to work in Virtualenv
```
$ workon opencv_cuda
```
## Object Detection

Run tkdnn, files will get palced into MOT16/test, make sure to configure config_visagx-brio.ini.
```
$ cd /usr/local/src/git/tkDNN
$ ./tkdnn-start-brio.sh
```

## Generate feature vectors
Will create the feature vectors for every file in the mot_dir.
```
$ cd deep_sort
$ python tools/generate_detections.py --model=resources/networks/mars-small128.pb --mot_dir=./MOT16/test --output_dir=./resources/detections/test
```

### Error: cannot allocate memory in static TLS block
OpenCV has a bug in which it sometimes cannot allocate memory for the first library loaded. E.g.:

```
Traceback (most recent call last):
  File "tools/generate_detections.py", line 7, in <module>
    import cv2
  File "/home/visadmin/.local/bin/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/cv2/__init__.py", line 181, in <module>
    bootstrap()
  File "/home/visadmin/.local/bin/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/cv2/__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
```
In this case:
export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
Permanent Change:
```
$ nano ~/.bashrc
```
add export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 at the bottom, save, exit
```
$ source ~/.bashrc
$ workon opencv_cuda
```

Afterwards generate_detections.py should run as expected.

## Send detections via Kafka
Configure send_detections.py to match the correct file names.
Run Startup_services.sh only on the main jetson.
Configure startup_services.sh, send detections and later deep_sort_app_multitracker.py to match the ip of the main jetson.
```
$ bash ../scripts/startup_services.sh
$ python tools/send_detections.py
```
This will write the detections into the topic "timed-images".

## Setup Calibration with MOMC Calibration
If already setup skip to Calibrate Camera
[MOMC repository](https://github.com/hlrs-vis/multi-object-multi-camera-tracker)

```
$ cd /usr/local/src/git
$ git clone git@github.com:hlrs-vis/multi-object-multi-camera-tracker.git
$ cd multi-object-multi-camera-tracker
$ git checkout deep_sort
$ workon opencv_cuda
$ pip install utm
$ pip install geojson
$ pip install matplotlib
```
## Calibrate Camera

### Intrinsic Calibration
Brio cameras are already calibrated.

### Distort Background Image
``` 
$ cd /usr/local/src/git/multi-object-multi-camera-tracker
$ python undistort_image.py -i  calibration/calibration02_i -f /usr/local/src/git/tkDNN/build/calibrationFrame0.jpg
```
### Create Object Points file
```
$ touch /usr/local/src/git/tkDNN/build/calibrationFrame0.yml
$ nano /usr/local/src/git/tkDNN/build/calibrationFrame0.yml
```
Manually edit the Points inside. To match the Object Points in World Coordinates. Schema:
```
%YAML:1.0
---
objp: !!opencv-matrix
   rows: 4
   cols: 3
   dt: d
   data: [ point1_x, point1_y, 0.,
        point2_x, point2_y, 0.,
        point3_x, point3_y, 0.,
        point4_x, point4_y, 0.
       ]
```
### Extrinsic Calibration
Click at calibration points in the same way the object points and then press enter.
```
$ python ImageCoordinatesTool2.py -p /usr/local/src/git/tkDNN/build/calibrationFrame0_undistorted.jpg
```
### Edit config
```
$ nano vhs-calibration/config_vhs.ini
```
Edit the following lines:
```
image_pts_path = /usr/local/src/git/tkDNN/build/calibrationFrame0_2024-04-11_1139.yml
object_pts_path = /usr/local/src/git/tkDNN/build/calibrationFrame0.yml
calibration_image_path = /usr/local/src/git/tkDNN/build/calibrationFrame0_undistorted.jpg
```
### Export the invHmat
```
$ python global_utm_calib.py -p vhs-calibration/config_vhs.ini
```
Saves the invHmat.
```
$ cd /usr/local/src/git/tkDNN/deep_sort
$ python tools/send_calibration.py
```
Sends the calibration matrix + the cam_id from tkdnn-start-brio.ini to topic "calibration".

# Setup the tracking site
Now working on receiving Jetson.

## Calibration 
When all cameras sent their calibration matrices.
```
$ cd /usr/local/src/git/tkDNN/deep_sort
$ python tools/get_calibration.py
```
Saves the calibration matrices as a .npy file.

## DeepSORT

```
$ python deep_sort_app_multicam.py
```

# Evaluation

```
$ pip install pandas
$ python tools/evaluate_multicam.py
```