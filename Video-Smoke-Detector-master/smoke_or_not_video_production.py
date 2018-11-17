'''
Video Smoke Detector

This program analyzes video frames to look for smoke.
It gives a confidence level score between 0% and 100%.
To speed up the analysis, the program only analyzes every 10th frame in the source video stream.
Thus for example, for a 30 fps video, the video is analyzed 3 times per second.
The program is based on the tensorflow retraining tutorial for flower recognition.
www.tensorflow.org/tutorials/image_retraining

Usage:
SOURCE_FILES is a list of videos need to be analyzed.
DESTINATION_FILES is a list of the desired output video file names.
The program overlays the analysis results at the bottom of the screen.
The original videos' audio tracks will be stripped from the output videos.


Author: Chen-Yi Liu
Date: September 3, 2017
'''

import tensorflow as tf
import cv2
import numpy as np
import math

SOURCE_FILES = ["video1.mp4",\
                "video2.mp4",\
                "video3.mp4"]
                
DESTINATION_FILES = ["output1.avi",\
                     "output2.avi",\
                     "output3.avi"]
                     
OUTPUT_VIDEO_DIMENSIONS = (720, 1280, 3)
METER_IMAGE = "meter.png"
RENEW_RATE = 1.0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
meter_image = cv2.imread(METER_IMAGE)


# Load the neural network
with tf.gfile.FastGFile("smoke_or_not.pb", 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')
  
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# start reading the source videos
for i in range(len(SOURCE_FILES)):
  video_input = cv2.VideoCapture(SOURCE_FILES[i])  
  video_output = cv2.VideoWriter(DESTINATION_FILES[i], fourcc, 30.0, (1280, 720))
  print("Analysing ", SOURCE_FILES[i])              
  
  if video_input.isOpened():
    # calculate the scaling factor to convert video of any dimensions to HD 720p
    # sacrafice the first frame of the video for this task
    ret, frame = video_input.read()
  
    aspect_ratio = frame.shape[1] / frame.shape[0]  
    scaling_factor = 0.0
    if aspect_ratio < OUTPUT_VIDEO_DIMENSIONS[1]/OUTPUT_VIDEO_DIMENSIONS[0]:
      scaling_factor = OUTPUT_VIDEO_DIMENSIONS[0] / frame.shape[0]
    else:
      scaling_factor = OUTPUT_VIDEO_DIMENSIONS[1] / frame.shape[1]
    
    half_height = int(frame.shape[0] * scaling_factor / 2)
    half_width = int(frame.shape[1] * scaling_factor / 2)
    new_height = half_height * 2
    new_width = half_width * 2
  
    # process video
    frame_counter = 0
    output_frame = np.zeros(OUTPUT_VIDEO_DIMENSIONS, dtype=np.uint8)
    confidence = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    max_confidence = 0.0
    prev_max_confidence = 0.0
    
    while(video_input.isOpened()):
      ret, frame = video_input.read()

      if ret == True:
       
        # run the smoke analysis
        if frame_counter % 10 == 0:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Tensorflow uses RGB whereas OpenCV uses BGR
          
          # full frame
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame})
          confidence[0] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[0]
          # the four quadrants
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame[0:int(frame.shape[0]/2), 0:int(frame.shape[1]/2), :]})
          confidence[1] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[1]
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame[int(frame.shape[0]/2):, 0:int(frame.shape[1]/2), :]})
          confidence[2] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[2]
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame[0:int(frame.shape[0]/2), int(frame.shape[1]/2):, :]})
          confidence[3] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[3]
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame[int(frame.shape[0]/2):, int(frame.shape[1]/2):, :]})
          confidence[4] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[4]
          # center quarter
          predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': frame[int(frame.shape[0]/4):int(frame.shape[0]*3/4),\
                                                                       int(frame.shape[1]/4):int(frame.shape[1]*3/4), :]})
          confidence[5] = RENEW_RATE * predictions[0] + (1.0 - RENEW_RATE) * confidence[5]
          prev_max_confidence = max_confidence
          # the confidence level is calculated as the average of 1) the value of full screen analysis 
          # AND 2) the maximum value of the four quadrants + central quarter analysis
          # this is to make the program more sensitive to smoke that is confined to a small region on the screen
          max_confidence = (max(confidence[1:]) + confidence[0]) / 2.0

          frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert back to the OpenCV BGR format

        # scale the image to 720p
        scaled_frame = cv2.resize(frame, (new_width, new_height))
        output_frame[360-half_height:360+half_height, 640-half_width:640+half_width, :] = scaled_frame

        # meter readings are interpolated at frames not analzed
        # the meter reading at every frame is delayed by 10 frames
        meter_reading = (frame_counter%10) / 10 * max_confidence + (1 - (frame_counter%10) / 10) * prev_max_confidence
        # put the analysis result on the meter
        output_frame[-200:, 440:840] |= meter_image
        cv2.putText(output_frame, "SMOKE", (585, 680), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 2)
        cv2.line(output_frame, (640, 700), (640 - int(120*math.cos(meter_reading*3.1416)),\
                                            700 - int(120*math.sin(meter_reading*3.1416))), (50, 255, 0), 4)

        # print on the console to show progress
        if frame_counter % 30 == 0:
          print(frame_counter, meter_reading)      

        # add the analyzed frame to the output video
        video_output.write(output_frame)
        frame_counter += 1
  
      else:
        break

  video_input.release()
  video_output.release()
  




  


  
