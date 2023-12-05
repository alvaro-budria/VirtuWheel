# VirtuWheel
### A 3D Driving Simulator with Body Pose Control


VirtuWheel is a 3D driving simulator that offers a unique experience by allowing users to control the steering wheel using their body pose. In contrast to traditional simulators that require specific hardware like a physical steering wheel and gear shift, our system only needs a regular laptop and a budget camera, and an optional Logitech command board for enhanced controllability. The simulator immerses users in a realistic real-world environment, enhancing the driving experience while making it more accessible.

![](./assets/img.png)

![](./assets/demo2.gif)

![](./assets/demo.gif)


## Features
* Body Pose Control: Users can control the steering wheel through their body pose.

* Integrated command board: It is possible to integrate a Logitech board into the system, allowing for additional actions like taking screenshots, changing the car speed, and changing the direction (forward/backward).

* Realistic Environment: The simulator provides a real-world driving experience with realistic scenery.

* Minimal Requirements: Only a regular laptop and a budget camera are required, making it accessible to a wide audience.


## Installation


### Python Real-time Pose Detector
Clone the repository and install the required dependencies:

`pip install opencv-python numpy>=1.24.0 mediapipe`

We changed some parameters that are hardcoded in Mediapipe's code base. Go to `solutions/drawing_utils.py` in Mediapipe's library (typically at `~/anaconda3/lib/python<VERSION>/site-packages/mediapipe/python/solutions/drawing_utils.py`), and set

```
_PRESENCE_THRESHOLD = 0.05
_VISIBILITY_THRESHOLD = 0.1
```

at the beginning of the file.

Then download the required model files (in a .zip) from [this Google Drive folder](https://drive.google.com/drive/folders/1USEdy_7uvwO4PIqsQJq8kT0sX4H4f7nn) and put the unzipped folder `OpenPose_models` under the home directory of the project.


### Unity 3D Environment

Download the 3D Unity environment from [this Google Drive folder](https://drive.google.com/drive/folders/1K-5VAzCNx5bGB1IafuC4UzgDrgbZ6NQs?usp=drive_link). Put it in the project home directory.
Refer to the Unity [official manual](https://docs.unity3d.com/Manual/) for installation instructions.


## Usage

First execute the Python pose detection backend and leave it running on a separate terminal.

`python obtain_angle.py`

Then run the project on Unity in game mode. Open the environment from Unity and play it as is.

Finally, enjoy.



## Authors

- **Malika Utailineyeva**. Developed the main Unity app and integrated the different components.

- **Jaume Ros**. Created the real-world Lausanne 3D environment.

- **Valentin Schepotev**. Programmed the Logitech command board and integrated it into the system.

- **Alvaro Budria**. Implemented the body pose detector and the steering wheel algorithm.
