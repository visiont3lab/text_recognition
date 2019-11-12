import cv2
import time
import numpy as np
from timeit import default_timer as timer
from detection import DETECTION
from ocrdetection import OCR
from transform import four_point_transform

""" For video in a folder """
#video_path = "output.avi"  # file video
video_path = 0  #webcam

# SETTING CAMERA
fps = 30
start_image = 0
vid = cv2.VideoCapture(video_path)
vid.set(cv2.CAP_PROP_FPS, fps)

if not vid.isOpened():
	raise IOError(("Couldn't open video file or webcam. If you're "
	"trying to open a webcam, make sure you video_path is an integer!")) 
		
vid_w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the codec and create VideoWriter object 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('output1.avi', fourcc, 8, (640, 480)) #8=fps os streaming

# skip images until reaching start_image
if start_image > 0:
	vid.set(cv2.CAP_PROP_POS_MSEC, start_image)
	
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()


# class construct
mydetection = DETECTION()
mydetection.load_model()

#OCR
myOCR = OCR()

while True:
	retval, image_opencv = vid.read()
	if not retval:
		print("Done!")
		break

	# INFERENCE #
	bboxes, polys = mydetection.test_net(image_opencv)

	## SQUARE BOXES
	for i in range(len(bboxes)):
		points = []
		for j in range(len(bboxes[i])):
			a = ( np.rint(bboxes[i][j][0]), np.rint(bboxes[i][j][1]) )
			if (j==(len(bboxes[i])-1)):
				b = ( np.rint(bboxes[i][0][0]), np.rint(bboxes[i][0][1]) )
			else:
				b = ( np.rint(bboxes[i][j+1][0]), np.rint(bboxes[i][j+1][1]) )
			cv2.line(image_opencv,a,b,(255,0,0),2)
			points.append(a)

		points = np.array(points, dtype='int32')
		roi = four_point_transform(image_opencv, points)
		img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# for k in enumerate(img):
			# cv2.imwrite('./zfolder/test' + str(i) + ".jpg", img)
		# save = cv2.imwrite('./zfolder/test.jpg',img)
		pred = myOCR.run(img)
		# cv2.imshow("roi-onebyone", roi)
		# cv2.waitKey(1)
		###############################
		cv2.putText(image_opencv, str(pred), (bboxes[i][0][0],bboxes[i][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # calculate fps
	curr_time = timer()
	exec_time = curr_time - prev_time
	prev_time = curr_time
	accum_time = accum_time + exec_time
	curr_fps = curr_fps + 1
	if accum_time > 1:
		accum_time = accum_time - 1
		fps = "FPS: " + str(curr_fps)
		curr_fps = 0
	# draw fps
	cv2.rectangle(image_opencv, (0,0), (50, 17), (255,255,255), -1)
	cv2.putText(image_opencv, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)			

	# IMSHOW
	cv2.imshow("Text detection", image_opencv)
	#out.write(image)

	#ESCAPE
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
out.release()
cv2.destroyAllWindows()
