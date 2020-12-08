##openCV to solve Harry Potter Cloak magic
##Dated 10-05-2019, Author:Mostafiz


##Import opencv(cv2), Numpy array(numpy) 
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
## Preparation for writing the ouput video
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0, (640,480))

##reading from the webcam 
IMG_WIDTH=256
IMG_HEIGHT=256
THRESHOLD = 0.4
EROSION = 1


def preprocess(img):
    im = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)

    if img.shape[0] >= img.shape[1]:
        scale = img.shape[0] / IMG_HEIGHT
        new_width = int(img.shape[1] / scale)
        diff = (IMG_WIDTH - new_width) // 2
        img = cv2.resize(img, (new_width, IMG_HEIGHT))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = img.shape[1] / IMG_WIDTH
        new_height = int(img.shape[0] / scale)
        diff = (IMG_HEIGHT - new_height) // 2
        img = cv2.resize(img, (IMG_WIDTH, new_height))

        im[diff:diff + new_height, :, :] = img
        
    return im



def postprocess(img_ori, pred):
	h, w = img_ori.shape[:2]

	mask_ori = (pred.squeeze()[:, :, 1] > THRESHOLD).astype(np.uint8)
	max_size = max(h, w)
	result_mask = cv2.resize(mask_ori, dsize=(max_size, max_size))

	if h >= w:
		diff = (max_size - w) // 2
		if diff > 0:
			result_mask = result_mask[:, diff:-diff]
	else:
		diff = (max_size - h) // 2
		if diff > 0:
			result_mask = result_mask[diff:-diff, :]
		
	result_mask = cv2.resize(result_mask, dsize=(w, h))

	# fill holes
	#     cv2.floodFill(result_mask, mask=np.zeros((h+2, w+2), np.uint8), seedPoint=(0, 0), newVal=255)
	#     result_mask = cv2.bitwise_not(result_mask)
	result_mask *= 255

	#     # erode image
	#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*EROSION + 1, 2*EROSION+1), (EROSION, EROSION))
	#     result_mask = cv2.erode(result_mask, element)

	# smoothen edges
	result_mask = cv2.GaussianBlur(result_mask, ksize=(9, 9), sigmaX=5, sigmaY=5)

	return result_mask



def overlay_transparent(background_img, img_to_overlay_t, mask, x, y, overlay_size=None):
    img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_RGB2RGBA)
    bg_img = background_img.copy()

    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
        
    ##print(mask.shape)
    ##print(bg_img.shape)

    mask = cv2.medianBlur(mask, 5)

    h, w, _ = img_to_overlay_t.shape
    #print(f"h: {h}, w: {w}")
    roi = bg_img[:h,:w]
    
    ##print(roi.shape)

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[:h, :w] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)

    return bg_img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 25)

count = 0
background = 0
model = load_model('unet_no_drop.h5')

## Capture the background in range of 60
#for i in range(60):
ret,background = cap.read()
#background = np.flip(background,axis=1)
cv2.imwrite("ple.jpg",background)
#print(f"This is background shape {background.shape}")
cv2.imshow("first",background)
#cv2.imwrite("oke.jpeg",background)
## Read every frame from the webcam, until the camera is open
while(cap.isOpened()):
	ret, img = cap.read()
	if not ret:
		break
	
	#img = np.flip(img,axis=1)
	img_ori = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
	img = preprocess(img)
	##print(img.shape)
	input_img = img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3)).astype(np.float32) / 255.
	#cv2.imshow("imshow",img_ori)
	pred = model.predict(input_img)
	mask = postprocess(img_ori, pred)
	##print(mask.shape)
	converted_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	invi = cv2.subtract(img_ori,converted_mask)
	##print(invi.shape)
	#cv2.imshow("converted_mask",invi)
	invi = cv2.subtract(invi,converted_mask)
	##print(invi.shape)
	#cv2.imshow("subtract",invi)
	##print(invi.shape)
	invi = cv2.cvtColor(invi, cv2.COLOR_BGR2RGB)
	#cv2.imshow("Let see our invi partition",invi)
	invi = cv2.subtract(background,invi)
	##print(background.shape)
	##print('\n\n\n')
	cv2.imshow("let see again",invi)
	#overlay = cv2.resize(invi,dsize=None,fx=0.4,fy=0.4)
	#mask_res = cv2.resize(mask,dsize=None,fx=0.4,fy=0.4)
	overlay_img = cv2.resize(invi, dsize=None, fx=1, fy=1)
	resized_mask = cv2.resize(mask, dsize=None, fx=1, fy=1)

	out_img = overlay_transparent(img_ori,invi,mask,0,0)
	cv2.imshow("let see again final",out_img)
    ## Generating the final output and writing
    #finalOutput = cv2.addWeighted(res1,1,res2,1,0)
    #out.write(finalOutput)
	# invi = overlay_transparent(background,invi,0,0)
	# cv2.imshow("let see final one",invi)
	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()	
cv2.destroyAllWindows()

