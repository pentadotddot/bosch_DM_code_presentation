import argparse
import time
from pathlib import Path
import math

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from scipy.spatial import distance
import socket
import binascii
from sklearn.cluster import KMeans




def IoU(boxA,boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxBArea)
	# return the intersection over union value
	return iou

def K_means(data,n_cluster):
	kmean=KMeans(n_clusters=n_cluster)
	kmean.fit(data)
	return kmean.cluster_centers_


def Zoom_y(y,y_min,y_max,t):
	if y<y_min:
		return ((1.0-t)/y_min)*y + t

	if y>y_min and y<y_max:
		return 1.0

	if y>y_max:
		return ((t-1)/y_max)*(y-2*y_max) + t


def superposition(vectors):

	return np.add(vectors)





def Calccenterofmass_ori(im0, cordVecPerson,massPerson):

	# calculate COM
	massSum = 0
	massWeightedSumX = 0
	massWeightedSumY = 0


	for k in range(len(cordVecPerson)):
		massSum += massPerson[k]
		massWeightedSumX += cordVecPerson[k][0]*massPerson[k]
		massWeightedSumY += cordVecPerson[k][1]*massPerson[k]
	X_avg = massWeightedSumX/massSum
	Y_avg = massWeightedSumY/massSum


	# calculate the average distance of bboxes from COM
	distances = []

	for l in range(len(cordVecPerson)):
		distances.append(calc_eucledian_dist([cordVecPerson[l][0],cordVecPerson[l][1]],[X_avg,Y_avg]) )

	avg_distance = np.average(np.array(distances))

	start = (int(X_avg),int(Y_avg) )
	cv2.circle(im0, start, int(2*avg_distance), (1,255,1), 2)


	# filter out bboxes which are further than the average distance multiplied by a constant, e.g. 2
	
	filtered_coords = []
	for m in range(len(cordVecPerson)):
		dist = calc_eucledian_dist([cordVecPerson[m][0],cordVecPerson[m][1]],[X_avg,Y_avg])
		if dist < 2*avg_distance:
			filtered_coords.append([cordVecPerson[m][0],cordVecPerson[m][1]]) 

	# recalculate COM with filtered outliers

	massSum = 0
	massWeightedSumX = 0
	massWeightedSumY = 0

	for k in range(len(filtered_coords)):
		massSum += massPerson[k]
		massWeightedSumX += filtered_coords[k][0]*massPerson[k]
		massWeightedSumY += filtered_coords[k][1]*massPerson[k]
	X_avg = massWeightedSumX/massSum
	Y_avg = massWeightedSumY/massSum

	return [int(X_avg),int(Y_avg)]


def Calccenterofmass_VCOM(coords,mass_ratio):

	massSum = 1 + 1.0/mass_ratio

	if len(coords[1]) > 0:
		if coords[1][0] + coords[1][1] > 0:
			massWeightedSumX = coords[0][0]*1 + coords[1][0]*(1.0/mass_ratio)
			massWeightedSumY = coords[0][1]*1 + coords[1][1]*(1.0/mass_ratio) 
			X_avg = massWeightedSumX/massSum
			Y_avg = massWeightedSumY/massSum

	else:
		X_avg,Y_avg = coords[0][0],coords[0][1]

	return [int(X_avg),int(Y_avg)]

def Gauss(x,mu,sigma):
	return math.exp(-((x-mu)/(sigma))**2)/(math.sqrt(math.pi)*sigma )


def z_test_outlier_velo(x,mu,sigma):

	if (x-mu)/(sigma/2)>1.0:
		return True

	else:
		return False 

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0) # only difference

def CalcStepCenterMove(coords,mass,X_original,Y_original,n_step,C,epsilon,D):


	time_b = time.time()
	massSum = len(coords)*mass
	massWeightedSumX = 0
	massWeightedSumY = 0

	for k in range(len(coords)):

		massWeightedSumX += coords[k][0]*mass
		massWeightedSumY += coords[k][1]*mass

	X_avg = massWeightedSumX/massSum
	Y_avg = massWeightedSumY/massSum


	r = math.sqrt( (X_avg-X_original)**2 + (Y_avg-Y_original)**2)

	S = C*math.exp(-r/D) + epsilon  # scale factor for smoothing

	X_step = (X_avg-X_original)/S
	Y_step = (Y_avg-Y_original)/S

	retVec = []
	Gauss_cumm = 0
	for k in range(int(n_step+1) ):
		#Gausss = Gauss(k,int((n_step+1)/2.0) ,int((n_step+1)/4.0))
		Gausss = 1.0/n_step
		Gauss_cumm += Gausss
		retVec.append([int(X_original+ Gauss_cumm*X_step), int(Y_original+ Gauss_cumm*Y_step)])     


	return retVec

def is_inside_box(centercoord, box):
	if centercoord[0] < box[2] and  centercoord[0] > box[0]:
		if centercoord[1] < box[3] and  centercoord[1] > box[1]:
			return True
		else:
			return False
	else:
		return False 

def calc_area_vec(coordsBall):
	areaVec = []
	for k in range(len(coordsBall)):
		areaVec.append( (coordsBall[k][0]+coordsBall[k][2])*(coordsBall[k][1]+coordsBall[k][3]) ) 

	areaVec = softmax(areaVec)

	return areaVec 

def compare_and_sort_coords(initialPersonCoords,newcoords,everyX):

	time_b = time.time()

	retVec = []

	newcordsLoop = list(range(len(newcoords)))

	for k in range(len(initialPersonCoords)):
		boxSize = 1.2*initialPersonCoords[k][2]

		
		refbox = [int(initialPersonCoords[k][0]-boxSize/2.0),int(initialPersonCoords[k][1]-boxSize/2.0), int(initialPersonCoords[k][0]+boxSize/2.0),int(initialPersonCoords[k][1]+boxSize/2.0) ]
		#IoUs = []
		button = 0
		oldArea = initialPersonCoords[k][2]*initialPersonCoords[k][3]



		for l in newcordsLoop:
			#newbox = [int(newcoords[l][0]-newcoords[l][2]/2.0),int(newcoords[l][1]-newcoords[l][3]/2.0),int(newcoords[l][0]+newcoords[l][2]/2.0),int(newcoords[l][1]+newcoords[l][3]/2.0)]
			start = [initialPersonCoords[k][0],initialPersonCoords[k][1]]
			end = [newcoords[l][0],newcoords[l][1]]
			delta_r = calc_eucledian_dist(start,end)
			

			center = [newcoords[l][0],newcoords[l][1]] 
			newArea = newcoords[l][2]*newcoords[l][3]



			#IoUs.append( [IoU(refbox,newbox),l])
			if is_inside_box(center, refbox) == True and abs(oldArea-newArea)/((oldArea+newArea)/2.0)<0.5 :
				
				#if delta_r < 15*everyX:
				retVec.append(newcoords[l])
				#newcordsLoop.pop(l)
				newcoords[l] = [0,0,0,0]
				button = 1
				break

		if button == 0:
			retVec.append([0,0,0,0])    


	return retVec

def plot_histogram(data1,data2):

	data1 = np.array(data1)
	df = pd.DataFrame(data1, columns =['vel_mini'])
	df.hist(column='vel_mini')
	df.plot()
	data2 = np.array(data2)
	df2 = pd.DataFrame(data2, columns =['vel_panorama'])
	df2.hist(column='vel_panorama')
	df2.plot()

	plt.show(block=False)
	plt.pause(30)
	#plt.hist(data, density=True, bins='auto')  # density=False would make counts
	#plt.ylabel('Probability')
	#plt.xlabel('Data');
	#plt.show()
	time.sleep(0.01)
	
	plt.close('all')



def calc_eucledian_dist(start,end):
	return math.sqrt( (end[0]-start[0])**2 + (end[1]-start[1])**2)



refPoints = np.array([

	#A, outer circle (anticlockwise)
	[50,    993], #1
	[1913, 1001], #2
	[3790,  992], #3
	[3191,  549], #4
	[2900,  329],#5 
	[2707,  187], #6
	[1919,  193], #7
	[1130,  187], #8
	[942,   331], #9
	[647,   551], #10

	#CC, center
	[1915,427] 
		

],np.float32)

refPoints_mini = np.array([
	#A, outer circle (anticlockwise)
	[28,  330], #1
	[305, 330], #2
	[582, 330], #3
	[582, 230], #4
	[582, 134],#5 
	[582,  32], #6
	[305,  32], #7
	[28 ,  32], #8
	[28 , 134], #9
	[28 , 230], #10
	
	#CC, center
	[305,180] 
		

],np.float32)

def dR_homography_rescale(start,end, minimapScale,transMatrix,status,refPoints,refPoints_mini):

	time_b = time.time()

	resCoords = []
	crocoords = []

	refPoints_mini = np.multiply(refPoints_mini,minimapScale)
	refPoints_mini = np.int_(refPoints_mini)

	
	imagepoint_s = [start[0],start[1],1]
	imagepoint_e = [end[0],end[1],1]
	
	worldpoint_s = np.dot(transMatrix,imagepoint_s)
	scalar = worldpoint_s[2]
	xworld_s = int((worldpoint_s[0]/scalar))
	yworld_s = int((worldpoint_s[1]/scalar)) #in 10demm 

	worldpoint_e = np.dot(transMatrix,imagepoint_e)
	scalar = worldpoint_e[2]
	xworld_e = int((worldpoint_e[0]/scalar))
	yworld_e = int((worldpoint_e[1]/scalar)) #in 10demm 


	return calc_eucledian_dist([xworld_s,yworld_s],[xworld_e,yworld_e])



def Point_homography_rescale(point, minimapScale,transMatrix,status,refPoints,refPoints_mini):



	resCoords = []
	crocoords = []

	refPoints_mini = np.multiply(refPoints_mini,minimapScale)
	refPoints_mini = np.int_(refPoints_mini)

	
	imagepoint_s = [point[0],point[1],1]
	
	
	worldpoint_s = np.dot(transMatrix,imagepoint_s)
	scalar = worldpoint_s[2]
	xworld_s = int((worldpoint_s[0]/scalar))
	yworld_s = int((worldpoint_s[1]/scalar)) #in 10demm 
	
	return [xworld_s,yworld_s]


def PerspTransfImage(cordVecPerson,ballcoords,imgW,imgH,transMatrix,status,minimapScale,refPoints,refPoints_mini):

	time_b = time.time()

	resCoords = []
	crocoords = []

	refPoints_mini = np.multiply(refPoints_mini,minimapScale)
	refPoints_mini = np.int_(refPoints_mini)

	for k in range(len(cordVecPerson)):
		imagepoint = [cordVecPerson[k][0], cordVecPerson[k][1], 1]
		w,h = cordVecPerson[k][2],cordVecPerson[k][3]
		worldpoint = np.dot(transMatrix,imagepoint)
		scalar = worldpoint[2]
		xworld = int((worldpoint[0]/scalar))
		yworld = int((worldpoint[1]/scalar)) #in 10demm 


		resCoords.append([xworld,yworld,w,h])

	for k in range(len(ballcoords)):
		w,h = ballcoords[k][2],ballcoords[k][3]
		imagepoint = [ballcoords[k][0], ballcoords[k][1], 1]
		worldpoint = np.array(np.dot(transMatrix,imagepoint))
		scalar = worldpoint[2]
		xworld = int((worldpoint[0]/scalar))
		yworld = int((worldpoint[1]/scalar)) #in 10demm 

		crocoords.append([xworld,yworld,w,h])



	'''
	for k in range(len(cordVecPerson)+1):
		resCoords[k][0][0] += imgW
		resCoords[k][0][1] += imgH  
	print(resCoords)
	'''
	return resCoords,crocoords

def return_index_to_skip(velocityVector, n_steps):
	index_to_skip = []
	for k in range(len(velocityVector) ):
		for l in range(n_steps):
			if np.prod(velocityVector[k][l]) == 0:
			   index_to_skip.append(k)
			   break

	return index_to_skip    


def draw_boxes_arrows(velocityVector, histvelvec, index_to_skip, im0, step_idx):
	for k in range(len(velocityVector)):
		if k in index_to_skip:
			continue
		else:     
			
			cv2.arrowedLine(im0, (velocityVector[k][step_idx-1][0],velocityVector[k][step_idx-1][1]),(velocityVector[k][step_idx][0],velocityVector[k][step_idx][1]) ,(221,111,111), 5)

			personVelo = histvelvec[k]
			if  personVelo > 10:
				cv2.rectangle(im0, ( int(velocityVector[k][step_idx][0]-velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]-velocityVector[k][step_idx][3]/2.0)),(int(velocityVector[k][step_idx][0]+velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]+velocityVector[k][step_idx][3]/2.0)) ,(0,0,255), 10)                                
			else:
				cv2.rectangle(im0, ( int(velocityVector[k][step_idx][0]-velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]-velocityVector[k][step_idx][3]/2.0)),(int(velocityVector[k][step_idx][0]+velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]+velocityVector[k][step_idx][3]/2.0)) ,(255,255,255), 7)                                  


			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (int(velocityVector[k][step_idx][0]-velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]-velocityVector[k][step_idx][3]/2.0))
			bottomLeftCornerOfText_2 = (int(velocityVector[k][step_idx][0]-velocityVector[k][step_idx][2]/2.0),int(velocityVector[k][step_idx][1]-velocityVector[k][step_idx][3]/2.0)-50)                                     
			fontScale              = 1.0
			fontColor              = (255,255,255)
			lineType               = 2
			cv2.putText(im0,str(round(personVelo,4) ),
											   bottomLeftCornerOfText, 
																	 font, 
															   fontScale,
															   fontColor,
																lineType)
def visualize_minimap(im0,velocityVector, coordsBall,MatrixT,status,minimapScale,refPoints,refPoints_mini, n_steps,imo_w,imo_h, index_to_skip, x_offset, y_offset, softVeloVec,cluster_velo_threshold,velocity_cluster_coordinates, frame_idx):
	
	minimap_x_min,minimap_x_max,minimap_y_min,minimap_y_max = 15,60,18,38
	minimap_w,minimap_h = int(612*minimapScale),int(364*minimapScale)


	
	if frame_idx%n_steps==0:

		data = PerspTransfImage(velocityVector[::][:][:,n_steps-1],coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)
		data_prev = PerspTransfImage(velocityVector[::][:][:,0],coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)
	else:

		data = PerspTransfImage(velocityVector[::][:][:,frame_idx%n_steps],coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)
		data_prev = PerspTransfImage(velocityVector[::][:][:,frame_idx%n_steps-1],coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)

	miniMapCoords = data[0]
	miniMapCoords_prev = data_prev[0]
	minimapBallCoords = data[1]


	# minimap bbox
	bbox_x1 = x_offset + minimap_x_min*minimapScale
	bbox_x2 = x_offset + minimap_w - minimap_x_max*minimapScale
	bbox_y1 = y_offset + minimap_y_min*minimapScale
	bbox_y2 = y_offset + minimap_h - minimap_y_max*minimapScale

	for circ in range(len(minimapBallCoords)):
		# ball coordinates to find if inside ROI
		cx = x_offset + int(minimapBallCoords[circ][0])
		cy = y_offset + int(minimapBallCoords[circ][1])
		if is_inside_box([cx,cy],[ bbox_x1 ,bbox_y1 ,bbox_x2 , bbox_y2]) == True:
			cv2.circle(im0, (cx,cy),10,(255,65,2),-1)
		


	velocity_cluster = []

	for circ in range(len(miniMapCoords)):

		if circ in index_to_skip:
			continue

		
		
		start = (x_offset + miniMapCoords_prev[circ][0],y_offset + miniMapCoords_prev[circ][1])
		end = (x_offset + miniMapCoords[circ][0],y_offset +miniMapCoords[circ][1])
		softVeloVec[circ] = calc_eucledian_dist([start[0],start[1]],[end[0],end[1]])
	

		if frame_idx%n_steps==0:
			cv2.arrowedLine(im0,  start, end ,(250,0,1), 5)
			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (end[0],end[1]-5)
			fontScale              = 0.75
			fontColor              = (10,10,255)
			lineType               = 1

		else:

			cv2.arrowedLine(im0,  start, end ,(0,0,251), 5)
			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = (end[0],end[1]-5)
			fontScale              = 0.75
			fontColor              = (10,10,255)
			lineType               = 1


		if softVeloVec[circ] > cluster_velo_threshold:

		  x_min_tmp = int(velocityVector[circ][-1][0])
		  y_min_tmp = int(velocityVector[circ][-1][1])
		  x_max_tmp = int(velocityVector[circ][-1][2])
		  y_max_tmp = int(velocityVector[circ][-1][3]) 
		  velocity_cluster_coordinates.append([x_min_tmp,y_min_tmp,x_max_tmp,y_max_tmp])

		velocity_cluster.append([end[0],end[1]])
		cv2.putText(im0,str(round(softVeloVec[circ],4)),
										 bottomLeftCornerOfText, 
															   font, 
														 fontScale,
														 fontColor,
														  lineType)

		cv2.arrowedLine(im0, start, end ,(221,111,111), 15)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (x_offset + int(miniMapCoords[circ][0]),y_offset + int(miniMapCoords[circ][1]-5) )
		fontScale              = 0.75
		fontColor              = (255,255,255)
		lineType               = 2
		
		cv2.putText(im0,str(round(calc_eucledian_dist([start[0],start[1]],[end[0],end[1]]),4) ),
									 bottomLeftCornerOfText, 
													   font, 
												  fontScale,
												  fontColor,
												   lineType)




def create_histvelvec(miniMapCoords, miniMapCoords_prev, index_to_skip, step_idx, x_offset,y_offset):

	histvelvec = [0]*50
	for circ in range(len(miniMapCoords)):

		if circ in index_to_skip:
			continue       
		if step_idx>0:
			start = (x_offset + miniMapCoords_prev[circ][0], y_offset + miniMapCoords_prev[circ][1])
			end = (x_offset + miniMapCoords[circ][0], y_offset + miniMapCoords[circ][1])

			velo = calc_eucledian_dist([start[0],start[1]],[end[0],end[1]])

			histvelvec[circ] = (velo)

	return histvelvec
		


minimapScale = 1.1

refPoints_mini =  np.multiply(refPoints_mini,minimapScale)
refPoints_mini =  np.int_(refPoints_mini)


MatrixT,status = cv2.findHomography(refPoints , refPoints_mini )


def detect(save_img=False):
	source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
		('rtsp://', 'rtmp://', 'http://'))

	# Directories
	save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Initialize
	set_logging()
	device = select_device(opt.device)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	vid_path, vid_writer = None, None
	if webcam:
		view_img = check_imshow()
		cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=imgsz, stride=stride)
	else:
		save_img = True
		dataset = LoadImages(source, img_size=imgsz, stride=stride)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
	t0 = time.time()

	# input parameters

	img_idx = 0
	frame_idx = 0
	doublefram_idx = 0
	everyX = 1
	cropScale = 1.0
	cropWindow_o = [cropScale* 440,cropScale*760] # size of camera
	cropWindow = cropWindow_o
	coordsPerson = []
	
	coordsBall = []

	appendingBool = 1
	

	cluster_velo_threshold = 9

	'''
	personCounter = 0
	ballCounter = 0
	maxPlayers = 10 + 1  # with referee
	crowdmass_scale_o  = 2     # weight of crowds mass
	crowdmass_scale = crowdmass_scale_o
	'''
	imo_w, imo_h = 3840, 1048
	im_o = []

	n_steps = 8  # number of steps for the coordinate calculation algo
	step_idx = 0 # index the steps
	epoch = 0 # an epoch stands for a n_step calculation and execution of the algo

	velocity_scale = 0.1  # the power of the velocity for weighting the person coordinates
	maxVelocity = 10.0  # if above 1 velocity weight is smaller if below 1 vhigher velocity players weigh more
	
	CameraCoords_execute = [[0,0]]*(n_steps) # here I store the coordinates to be executed onscreen
	CameraCoords_append = [[0,0]]*(n_steps)  # here I store the coordinates to append for the next

	CameraCoords_execute_VCOM = [[0,0]]*(n_steps) # here I store the coordinates to be executed onscreen
	CameraCoords_append_VCOM = [[0,0]]*(n_steps)  # here I store the coordinates to append for the next


	velocityVector = np.array([[[0,0,0,0]]*n_steps]*50) # here I predefine the vector where I will store the players velocities (50 is a massive overestimate of maximum detected persons)
	velocityVector_VCOM = np.array([[[0,0,0,0]]*n_steps]*50) # here I predefine the vector where I will store the players velocities (50 is a massive overestimate of maximum detected persons)
	velocityVector_minimap = np.array([[[0,0,0,0]]*n_steps]*50) # here I predefine the vector where I will store the players velocities (50 is a massive overestimate of maximum detected persons)
	initialPersonCoords = []  # first batch of people with which coordinates I will compare future ones

	lastCoords = [int(imo_w/2.0) , int(imo_h/2.0)] # initialize lastcoordinate to the center of the screen
	lastCoords_minimap = [int(imo_w/2.0) , int(imo_h/2.0)] # initialize lastcoordinate to the center of the screen  
	lastCoords_stopped = [int(imo_w/2.0) , int(imo_h/2.0)] # initialize lastcoordinate_stopped (non updating lastcoords) to the center of the screen
	lastPersonCoords = []
	lastCoordsVCOM = []

	softVeloVec = [0]*len(velocityVector)

	velocity_cluster_coordinates = []
	velocity_cluster_coordinates_step = []

	velocity_cluster_center = []
	velocity_cluster_center_prev = []
	velocity_cluster_center_step = []

	massPersonVec = []
	massCrowdVec = []

	histvelvec = []
	truVelVec = []

	miniMapCoords = []

	data_mini = [0]*len(velocityVector)
	break_counter = 0

	minimap_w,minimap_h = int(612*minimapScale),int(364*minimapScale)
	s_img = np.array(cv2.imread("basketball_court.jpg"))
	s_img = cv2.resize(s_img,(minimap_w,minimap_h))

	running = True

	for path, img, im0s, vid_cap in dataset:

		if running == False:
			break
		#if break_counter == 10:
		#   break

		key = cv2.waitKey(1)  # 1 millisecond
		
		t_z = time.time()
		coordsPerson = []
		coordsPerson_prev = []
		coordsBall = []
		coordsBall_prev = []

		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)
			
		# Inference
		t_b = time.time()
		t1 = time_synchronized()
		pred = model(img, augment=opt.augment)[0]

		# Apply NMS
		pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		personCounter = 0
		ballCounter = 0

		# Process detections
		for i, det in enumerate(pred):  # detections per image
			
			if webcam:  # batch_size >= 1
				p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
			else:
				p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

			im0 = cv2.resize(im0,(imo_w,imo_h))

			im_o = im0.copy()
			x_offset,y_offset=int(imo_w-minimap_w),0
			im0[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img


			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # img.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
			s += '%gx%g ' % img.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				time_b = time.time()
				for *xyxy, conf, cls in reversed(det):
					if view_img:  # Write to file
						

						xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
						
						
						line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
						if line[2]*imo_h < 140:
							continue
						print(names[int(cls)])
						if f'{names[int(cls)]}' == 'person':
							coordsPerson.append([line[1]*imo_w,line[2]*imo_h,line[3]*imo_w,line[4]*imo_h])
							personCounter +=1

						if f'{names[int(cls)]}' == 'sports ball':
							coordsBall.append([line[1]*imo_w,line[2]*imo_h,line[3]*imo_w,line[4]*imo_h])
							if line[3]*imo_w*line[4]*imo_h > 0:
								plot_one_box(xyxy, im0, label="BALL", color=colors[0], line_thickness=3)
				
								ballCounter +=1
						
					if save_img or view_img:  # Add bbox to imaqge
						label = f'{names[int(cls)]} {conf:.2f}'
			

			try:
				COM = Calccenterofmass_ori(im0,coordsPerson,[1]*len(coordsPerson))

				cv2.circle(im0, (int(COM[0]),int(COM[1]) ),311,(0,0,255),5)

			except Exception as e:
				print(e)
				pass
						
			#if len(coordsPerson)+len(coordsBall) == 0:
			#   continue

			t_b = time.time()
			time.sleep(0.001) #overhead for live frames

				
			if frame_idx%n_steps== 0 and frame_idx > 0:
				
				step_idx = 0                
				
				epoch +=1 # epoch ends
				
				velocity_cluster_center_prev = velocity_cluster_center.copy()

				
				index_to_skip = []
				index_to_include_velo = []
				truVelVec = []
				
				index_to_skip = return_index_to_skip(velocityVector, n_steps) # some bboxes getting lost, thus tracking can become unprecise, thus I just remove those tracks (could be more elegant)

				
				# draw an arrow from the starting point of a track to the end in a full epoch
				for k in range(len(velocityVector)):

					if k in index_to_skip:
						continue
					else:   
						cv2.arrowedLine(im0, (velocityVector[k][0][0],velocityVector[k][0][1]),(velocityVector[k][n_steps-1][0],velocityVector[k][n_steps-1][1]) ,(121,111,211), 10)

						#cv2.rectangle(im0, ( int(velocityVector[k][n_steps-1][0]-velocityVector[k][n_steps-1][2]/2.0),int(velocityVector[k][n_steps-1][1]-velocityVector[k][n_steps-1][3]/2.0)),(int(velocityVector[k][n_steps-1][0]+velocityVector[k][n_steps-1][2]/2.0),int(velocityVector[k][n_steps-1][1]+velocityVector[k][n_steps-1][3]/2.0)) ,(255,255,255), 10)
						truVelVec.append(math.sqrt(abs(velocityVector[k][0][0]-velocityVector[k][n_steps-1][0])**2 + abs(velocityVector[k][0][1]-velocityVector[k][n_steps-1][1])**2))
			
				
				
				# invert the homography matrix for inverting the coordinates back to basketball pitch
				inv_homo_matrix = np.linalg.inv(MatrixT)

				# transform bbox coordinates to eagle eye view
				visualize_minimap(im0,velocityVector, coordsBall,MatrixT,status,minimapScale,refPoints,refPoints_mini, n_steps,imo_w,imo_h,index_to_skip, x_offset, y_offset, softVeloVec,cluster_velo_threshold,velocity_cluster_coordinates, frame_idx)

				

	

			if img_idx%everyX != 0:
				img_idx+=1
				continue

			else:

				if step_idx == 0:

				
					initialPersonCoords = coordsPerson # first step I initialize person coordinates with the last batch last element
					#print(velocityVector)
					# store elements in the main array
					for k in range(len(initialPersonCoords)):  # copy new coords to main vector
						velocityVector[k][step_idx] = initialPersonCoords[k]
					
				else:

					initialPersonCoords =  velocityVector[::][:][:,step_idx-1]
					newcoordsTMP = compare_and_sort_coords(initialPersonCoords,coordsPerson ,everyX)
					
					velocityVector[::][:][:,step_idx] = newcoordsTMP					

					# homography small eagle eye map
					if len(coordsBall) == 0:
						coordsBall.append([0,0,0,0])
					#s_img = cv2.warpPerspective(im0,MatrixT,(int(612*minimapScale),int(364*minimapScale) ))

					coordsPerson = velocityVector[::][:][:,step_idx]
					data = PerspTransfImage(coordsPerson,coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)
					
					miniMapCoords = data[0]					
					minimap_x_min,minimap_x_max,minimap_y_min,minimap_y_max = 15,60,18,38

					coordsPerson_prev = velocityVector[::][:][:,step_idx-1]
					data_prev = PerspTransfImage(coordsPerson_prev,coordsBall,imo_w,imo_h, MatrixT,status,minimapScale,refPoints,refPoints_mini)
					miniMapCoords_prev = data_prev[0]
	
					index_to_skip = return_index_to_skip(velocityVector, n_steps)

					histvelvec = create_histvelvec(miniMapCoords, miniMapCoords_prev,index_to_skip, step_idx, x_offset, y_offset)

					#calculate avg velocity for thresholding
					vel_avg = np.average(histvelvec)
					vel_sigma = np.std(histvelvec)

					draw_boxes_arrows(velocityVector, histvelvec, index_to_skip,im0, step_idx)

					visualize_minimap(im0,velocityVector, coordsBall,MatrixT,status,minimapScale,refPoints,refPoints_mini, n_steps,imo_w,imo_h,index_to_skip, x_offset, y_offset, softVeloVec,cluster_velo_threshold,velocity_cluster_coordinates, frame_idx)

	
		
			frame_idx +=1
			step_idx +=1
			img_idx+=1

			# Stream results
			if view_img:
				try:
	
					font                   = cv2.FONT_HERSHEY_SIMPLEX
					bottomLeftCornerOfText = (0,100)
					fontScale              = 2
					fontColor              = (255,255,255)
					lineType               = 2

					for circ in range(len(miniMapCoords)):
						if np.prod(miniMapCoords[circ])==0:
							continue
					
						cv2.circle(im0, (x_offset + int(miniMapCoords[circ][0]),y_offset + int(miniMapCoords[circ][1]) ),5,(0,0,255),-1)


					cv2.putText(im0,"FPS: "+str(round(1.0/(time.time()-t_z),2) ), 

					bottomLeftCornerOfText, 
					font, 
					fontScale,
					fontColor,
					lineType)



					print(time.time()-t_z)
					if epoch>1:
						cv2.imshow(str(p), cv2.resize(im0,(int(imo_w/3),int(imo_h/3)) ) )

					coordsPerson = []
					coordsBall = []

					#print("time:",time.time()-t_z)
				except Exception as e:
					print("2",e)

				#im0 = im0[500:cropWindow[1],500:cropWindow[0]]
							

		
			# Save results (image with detections)
			
			
			if save_img:
				if dataset.mode == 'image':
					cv2.imwrite(save_path, im0)
				   # video 
				fps = vid_cap.get(cv2.CAP_PROP_FPS) 
				w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
				h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
				if vid_cap:  # 'video'
					if vid_path != save_path:  # new video
						vid_path = save_path
						if isinstance(vid_writer, cv2.VideoWriter):
							vid_writer.release()  # release previous video writer

						fourcc = 'mp4v'  # output video codec
						fps = vid_cap.get(cv2.CAP_PROP_FPS)
						w = imo_w
						h = imo_h
						vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
					vid_writer.write(im0)
				else:  # stream 
					fps, w, h = 25.0, im0.shape[1], im0.shape[0] 
					save_path += '.mp4' 
					vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)) 
					vid_writer.write(im0)

			if key & 0xFF == ord("q"):
				
				running = False
				
				plot_histogram(np.divide(histvelvec,minimapScale*1023) ,np.divide(truVelVec,3840) )


	if save_txt or save_img:
		s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=2000, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	opt = parser.parse_args()
	
	check_requirements()

	with torch.no_grad():
		while(True):
				
			if opt.update:  # update all models (to fix SourceChangeWarning)
				for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
					detect()
					strip_optimizer(opt.weights)
			else:
				detect()



#time.sleep(0.5)
cv2.destroyAllWindows()