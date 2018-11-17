# Video-Smoke-Detector

This program analyzes video frames to look for smoke.
It gives a confidence level score between 0% and 100%.
To speed up the analysis, the program only analyzes every 10th frame in the source video stream.
Thus for example, for a 30 fps video, the video is analyzed 3 times per second.
The program is written based on the tensorflow retraining tutorial for flower recognition.
www.tensorflow.org/tutorials/image_retraining


### Usage:
The program takes in a list of video files and overlays the smoke analysis results as a meter reading at the bottom of the screen.
The original videos' audio tracks will be stripped from the output videos.

### Examples:

![Sample result 1](https://github.com/chenyi731231/Video-Smoke-Detector/blob/master/sample1.png)

![Sample result 2](https://github.com/chenyi731231/Video-Smoke-Detector/blob/master/sample2.png)

Examples of analyzed videos can be viewed on YouTube.

<a href="https://www.youtube.com/watch?v=LekBzCjBpww
" target="_blank"><img src="http://img.youtube.com/vi/LekBzCjBpww/0.jpg" 
alt="Video Smoke Detection" width="480" height="360" border="10" /></a>
