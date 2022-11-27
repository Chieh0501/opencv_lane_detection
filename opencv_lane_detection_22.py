import cv2 as cv2 #importing the library
import numpy as np
import matplotlib.pyplot as plt
import time

def show_image(name,img): #function for displaying the image
    cv2.imshow(name,img)
    cv2.waitKey(200)
    cv2.destroyAllWindows()

def find_canny(img,thresh_low,thresh_high): #function for implementing the canny
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur,thresh_low,thresh_high)
    # show_image('Canny',img_canny)
    return img_canny

def region_of_interest(image): #function for extracting region of interest
    #bounds in (x,y) format
    # bounds = np.array([[[0,250],[0,200],[150,100],[500,100],[650,200],[650,250]]],dtype=np.int32)
    bounds = np.array([[[0,500],[0,350],[650,350],[650,500]]],dtype=np.int32)
    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,bounds,[255,255,255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image,mask)
    # show_image('mask',masked_image) 
    return masked_image


def draw_lines(img,lines): #function for drawing lines on black mask
    mask_lines=np.zeros_like(img)
    for points in lines:
        x1,y1,x2,y2 = points[0]
        cv2.line(mask_lines,(x1,y1),(x2,y2),[0,0,255],2)

    return mask_lines

def get_coordinates(img,line_parameters): #functions for getting final coordinates
    slope=line_parameters[0]
    intercept = line_parameters[1]
    y1 =300
    y2 = 120
    # y1=img.shape[0]
    # y2 = 0.6*img.shape[0]
    x1= int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return [x1,int(y1),x2,int(y2)]

def compute_average_lines(img,lines):
    left_lane_lines=[]
    right_lane_lines=[]
    left_weights=[]
    right_weights=[]
    for points in lines:
        x1,y1,x2,y2 = points[0]
        if x2==x1:
            continue     
        parameters = np.polyfit((x1,x2),(y1,y2),1) #implementing polyfit to identify slope and intercept
        slope,intercept = parameters
        length = np.sqrt((y2-y1)**2+(x2-x1)**2)
        if slope <0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)         
        else:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)
    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines,axis=0)
    right_average_line = np.average(right_lane_lines,axis=0)
    print(left_average_line,right_average_line)
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img,left_average_line)
    right_fit_points = get_coordinates(img,right_average_line) 
    # print(left_fit_points,right_fit_points)
    return [[left_fit_points],[right_fit_points]] #returning the final coordinates


##Implementation

# lane_canny = find_canny(result,100,200)
# lane_roi = region_of_interest(lane_canny)
# lane_lines = cv2.HoughLinesP(lane_roi,1,np.pi/180,50,40,5)
# # print(lane_lines)
# lane_lines_plotted = draw_lines(result,lane_lines)
# show_image('lines',lane_lines_plotted)
# result_lines = compute_average_lines(result,lane_lines)
# print(result_lines)
# final_lines_mask = draw_lines(result,result_lines)
# show_image('final',final_lines_mask)

# for points in result_lines:
#     x1,y1,x2,y2 = points[0]
#     cv2.line(result,(x1,y1),(x2,y2),(0,0,255),5)

# show_image('output',result)






# Video Processing:
# cap = cv2.VideoCapture(0)
# if not cap.isOpened:
#     print('Error opening video capture')
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print(' No captured frame -- Break!')
#         break
#     result = np.copy(frame)
#     lane_canny = find_canny(result,100,200)
#     show_image('canny',lane_canny)
#     # lane_roi = region_of_interest(lane_canny)
#     # lane_lines = cv2.HoughLinesP(lane_roi,1,np.pi/180,50,40,5)
#     # lane_lines_plotted = draw_lines(result,lane_lines)
#     # # show_image('lines',lane_lines_plotted)
#     # result_lines = compute_average_lines(result,lane_lines)
#     # # print(result_lines)
#     # final_lines_mask = draw_lines(result,result_lines)
#     # # show_image('final',final_lines_mask)
#     # for points in result_lines:
#     #     x1,y1,x2,y2 = points[0]
#     #     cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
#     # show_image('output',frame)



cap = cv2.VideoCapture('output26.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # cv2.imshow('Frame',frame)
    image = frame #Reading the image file
    result = np.copy(image)
    result = cv2.resize(result,(650,500))
    result = cv2.GaussianBlur(result,(9,9),0)
    lower_yellow = np.array([20,43,100])
    upper_yellow = np.array([42,255,255])
    result = region_of_interest(result)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = cv2.bitwise_and(result, result, mask = mask)    
    lane_canny = find_canny(result,100,200)
    # cv2.imshow('canny',lane_canny)
    # cv2.waitKey(200)
    # cv2.destroyAllWindows()
    # # show_image('canny',lane_canny)
    # lane_roi = region_of_interest(lane_canny)
    lane_lines = cv2.HoughLinesP(lane_canny,1,np.pi/180,30,40,5)
    print(lane_lines)
    if lane_lines is not None :
      lane_lines_plotted = draw_lines(result,lane_lines)
      show_image('lines',lane_lines_plotted)
      print("Here mothafucka")
      result_lines = compute_average_lines(result,lane_lines)
      
      print(lane_lines[1])
      print(result_lines)

    # final_lines_mask = draw_lines(result,result_lines)
    # show_image('final',final_lines_mask)

    # for points in result_lines:
    #     x1,y1,x2,y2 = points[0]
    #     cv2.line(result,(x1,y1),(x2,y2),(0,0,255),5)

    # show_image('output',result)

    # lower_blue = np.array([50,100,0])
    # upper_blue = np.array([255,255,255])
    # mask = cv2.inRange(cropped, lower_blue, upper_blue)
    # res = cv2.bitwise_and(cropped,cropped, mask= mask)
    # cv2.imshow('blur',res)
    # cv2.waitKey(200)
    # cv2.destroyAllWindows()
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


# image = cv2.imread('1.png') #Reading the image file
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# result = np.copy(image)
# result = cv2.resize(result,(650,500))
# cropped = result[350:500, 0:650]

# lower_yellow = np.array([25,60,50])
# upper_yellow = np.array([42,150,80])
# hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# cv2.imshow('yellow_filter',mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# result = cv2.bitwise_and(cropped, cropped, mask = mask)
# cv2.imshow('filtered',result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # lower_blue = np.array([50,100,0])
# upper_blue = np.array([255,255,255])
# mask = cv2.inRange(cropped, lower_blue, upper_blue)
# res = cv2.bitwise_and(cropped,cropped, mask= mask)
# cv2.imshow('cropped',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# lane_gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',lane_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# lane_blur = cv2.GaussianBlur(res,(5,5),0)
# cv2.imshow('blur',lane_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# lane_canny = cv2.Canny(res,180,200)
# cv2.imshow('canny',lane_canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# lowerThreshold = 44
# higherThreshold = 255
# returnValue, binaryThresholdedImage = cv2.threshold(vChannel,lowerThreshold,higherThreshold,cv2.THRESH_BINARY)
# thresholdedImage = cv2.cvtColor(binaryThresholdedImage, cv2.COLOR_GRAY2RGB)
# cv2.imshow('final',thresholdedImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
