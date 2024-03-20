import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def detect_pupil(image):
    b,g,r = cv.split(image)
    g_blurred = cv.GaussianBlur(g,(43,43),7)
    circles = cv.HoughCircles(g_blurred,cv.HOUGH_GRADIENT,1,10,
                            param1=15,param2=20,minRadius=20,maxRadius=40)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     return circles[0][0]
    
    return circles

def detect_iris(image):
    b,g,r = cv.split(image)
    r_blurred = cv.GaussianBlur(b,(21,21),3)
    circles = cv.HoughCircles(r_blurred,cv.HOUGH_GRADIENT,1,50,
                            param1=50,param2=25,minRadius=95,maxRadius=140)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]
    
    return circles
    

if __name__ == "__main__":
    image = cv.imread("14-1.tiff")
    # cv.imshow("Image1",image)

    b,g,r = cv.split(image)


    # g_blurred21 = cv.GaussianBlur(r,(21,21),3)
    g_blurred27 = cv.GaussianBlur(r,(27,27),4)
    g_blurred35 = cv.GaussianBlur(r,(35,35),5)
    g_blurred43 = cv.GaussianBlur(r,(43,43),6)

    # # Canny Detection Test
    cannyT = 50
    # edges21 = cv.Canny(g_blurred21,cannyT,cannyT/2)
    edges27 = cv.Canny(g_blurred27,cannyT,cannyT/2)
    edges35 = cv.Canny(g_blurred35,cannyT,cannyT/2)
    edges43 = cv.Canny(g_blurred43,cannyT,cannyT/2)
    plt.subplot(221),plt.imshow(image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(edges27,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(edges35,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(edges43,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Circle detection test
    circles = cv.HoughCircles(g_blurred43,cv.HOUGH_GRADIENT,1,40,
                            param1=cannyT,param2=20,minRadius=80,maxRadius=110)

    # circles = cv.HoughCircles(g_blurred,cv.HOUGH_GRADIENT_ALT,1,5,
    #                         param1=54,param2=0.8,minRadius=0,maxRadius=0)
    
    cimg = cv.cvtColor(g,cv.COLOR_GRAY2BGR)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        print(i)
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    # circle = detect_iris(image)
    # if circle is not None:
    #     print(circle)
    #     # draw the outer circle
    #     cv.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv.circle(image,(circle[0],circle[1]),2,(0,0,255),3)

    cv.imshow("Green Layer", cimg)

    cv.waitKey(0)
    cv.destroyAllWindows()
