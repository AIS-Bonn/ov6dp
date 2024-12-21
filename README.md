# Open Vocabulary 6D Pose Estimation (OV6DP)

This repository contains the code for the OV6DP project. 

## Installation

This module requires [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) to be installed first as the backbone for object recognition. Next, further requirements can be installed through the `requirements.txt` file. 

## Usage

Sample usage of the modules functionalities are presented in the main method of [ov6dp.py](ov6dp/ov6dp.py). There, it is shown what the correct image format is, how to pass it into the model, and what outputs are returned.

Note, that the module expects object model files to be located in a `models` folder located inside this repository. This can for example be achived through a softlink to the actual model file location on your system.

For the ARMAR robot of the KIT the file [armar.py](ov6dp/armar.py) contains code to execute the module on the robot.

When code is executed it should be run from the `ov6dp` directory.