# bosch_DM_code_presentation
A presentation for TEAM FIT by Demeter Márton


In order to use my demo code, please clone the ultralytics yolov5 repo: https://github.com/ultralytics/yolov5

Then add all the files in this repo to the main directory of yolov5 including basketball_court.jpg and the test video under this url:

https://drive.google.com/file/d/1GlujJwIzzCy8QECYuAznldbU5Tv2D40Z/view?usp=sharing

Please also download the original yolo weigths, more specifically the yolov5x.pt file and add to the main directory.

To execute the demo, run the replaced detect.py file the following way from terminal (tested on ubuntu 18):

`python detect.py --source ./basket_test.mp4 --weights ./yolov5x.pt --view-img --save-txt`

Check out the presentation slides here: https://drive.google.com/file/d/1Fj21gLpHQZLQABRWDE8Fq3hisCTFgoYe/view?usp=sharing

DEMO video: https://youtu.be/rUhcwkdqvl8
<video src=https://youtu.be/rUhcwkdqvl8 width=180/>

If there is any question please write to: demeter.marton.cs@gmail.com
 
