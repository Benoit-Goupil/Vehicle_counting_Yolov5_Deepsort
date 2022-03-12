# Vehicle-counting-Yolov5-DeepSort

The detection part is based on the work of  Mikel Broström/https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch

## Project objectives

The objective of this project is to count the number of vehicles passing on traffic lanes. It uses the Yolov5 algorithm for object detection and classification (on the examples below we differentiate cars and trucks) and deepsort for vehicle reidentification from one frame to another. 

In the examples below, blue is the car count and green is the truck count.


https://user-images.githubusercontent.com/73244633/158021027-3b11db88-0dca-44d3-8ab8-4621cc21e377.mp4






https://user-images.githubusercontent.com/73244633/158021039-52215d91-41e0-4735-bbe6-77798fe104ff.mp4




## How to use it 

1. Install all the requirements with the requirements.txt file

2. Add the video files of the cam you want to analyse in the parent file

3. You can execute the two examples configured for cam9 and cam10 of the AI city challenge dataset with the commands :

      ```bash
      python track_cam9.py --source cam_9.mp4 --show-vid --save-txt --imgsz 1920 1080 --save-vid --classes 2 3 5 7
      ```
      ```bash
      python track_cam10.py --source cam_10.mp4 --show-vid --save-txt --imgsz 1920 1080 --save-vid --classes 2 3 5 7 
      ```

	- source option is the path of the video
	- show-vid option is to display the video
	- save-txt option is to save the logs of the whole experiment
	- imgsz option is the size of the image
	- save-vid is to save the final video analysed and annotated 
		- interesting to activate because the direct analysing process can take a long time 
	- classes options is to restrict the amount of classes detected by Yolov5 (important to put to 2 3 5 7 or less)
		- 2: cars
		- 3: motorcycles
		- 5: bus
		- 7: trucks

4. If you want to apply this script to another cam:
	- You need to configure the coordinates of your masks in the script in line 112-114 depending the number of masks
	you want to put. 
	- You need to adjust the threshold depending the framerate and the configuration of your camera
	- If you want a cool display of the number of cars and trucks, you can adjust them in line 234-253
	- Finally, be careful of the direction of your masks by adjusting the orientation of the projection of 
	the speed vector 
