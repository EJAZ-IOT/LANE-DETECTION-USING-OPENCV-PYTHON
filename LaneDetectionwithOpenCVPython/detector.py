#import matplotlib.pylab as plt
import cv2
import numpy as np


def region_of_interest(img, vertices):           #creating masked image using ROI
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):                             #draw lines using coordinates from hough transform
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0,255, 0), thickness=3)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

#image = cv2.imread('road.png')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [                   #in this example triangular shape is used as ROI
        (0, height),                                  #initial point
        (width /2, height/2),                         #mid point
        (width, height)                               #end point
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     #converted to gray
    canny_image = cv2.Canny(gray_image, 80, 120)             #edge detection using using Canny
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32))  #cropping frame

    lines = cv2.HoughLinesP(cropped_image,                   #hough line transform probation
                            rho=1,                          # rho value every pixel
                            theta=np.pi / 180,               #theta as 180 radian
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('test_video.mp4')

#if you want to save the output

#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
#size = (frame_width, frame_height)
#result = cv2.VideoWriter('Output.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    #result.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
#result.release()
cv2.destroyAllWindows()

