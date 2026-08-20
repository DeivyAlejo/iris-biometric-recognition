import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

def detect_pupil(image):
    b,g,r = cv.split(image)

    layer = r

    low, high = np.percentile(layer, (1, 99))
    
    # Normalize and clip to [0, 1]
    gray_norm = np.clip(
        (layer.astype(np.float32) - low) / (high - low),
        0,
        1
    )

    # Apply gamma = 0.5
    grayp = np.power(gray_norm, 0.5)

    # Convert back to uint8 [0, 255]
    layer = (grayp * 255).astype(np.uint8)

    r_blurred = cv.GaussianBlur(layer,(21,21),3)
    cv.imshow("Red Layer", r_blurred)
    circles = cv.HoughCircles(r_blurred,cv.HOUGH_GRADIENT,1,10,
                            param1=25,param2=20,minRadius=10,maxRadius=45)

    if circles is not None:
        image_height, image_width = image.shape[:2]
        image_center = np.array([image_width / 2, image_height / 2])
        candidates = circles[0]
        distances_squared = np.sum(
            (candidates[:, :2] - image_center) ** 2,
            axis=1,
        )
        closest_circle = candidates[np.argmin(distances_squared)]
        return np.uint16(np.around(closest_circle))

    return None

def detect_iris(image):
    b,g,r = cv.split(image)

    layer = r

    low, high = np.percentile(layer, (1, 99))

    grayi = np.clip((layer.astype(np.float32) - low) / (high - low), 0, 1)
    grayi = (grayi * 255).astype(np.uint8)

    r_blurred = cv.GaussianBlur(grayi,(43,43),7)
    circles = cv.HoughCircles(r_blurred,cv.HOUGH_GRADIENT,1,50,
                            param1=25,param2=30,minRadius=90,maxRadius=150)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]
    
    return circles
    

if __name__ == "__main__":
    # # Test the detect_pupil function on a sample image
    image = "iris_detection_results/1/1.jpg"
    img = cv.imread(image)
    b,g,r = cv.split(img)
    
    layer = r

    low, high = np.percentile(layer, (1, 99))
    
    # Normalize and clip to [0, 1]
    gray_norm = np.clip(
        (layer.astype(np.float32) - low) / (high - low),
        0,
        1
    )

    # Apply gamma = 0.5
    grayp = np.power(gray_norm, 0.5)

    # Convert back to uint8 [0, 255]
    layer = (grayp * 255).astype(np.uint8)

    r_blurred = cv.GaussianBlur(layer,(21,21),3)
    cv.imshow("Red Layer", r_blurred)
    circles = cv.HoughCircles(r_blurred,cv.HOUGH_GRADIENT,1,10,
                            param1=25,param2=15,minRadius=10,maxRadius=45)

    if circles is not None:
        for circle in circles[0,:]:
            x, y, r = map(int, circle)
            # outer circle
            cv.circle(img, (x, y), r, (0, 255, 0), 2)
            # center point
            cv.circle(img, (x, y), 2, (0, 0, 255), -1)
    else:
        print("No pupil circles detected")

    cv.imwrite("iris_detection_results/1/1_pupil.jpg", img)


    # Test to detect the iris on a sample image
    # image = "iris_detection_results/3/8.jpg"
    # img = cv.imread(image)
    # # circles_pupil = detect_pupil(img)

    # # if circles_pupil is not None:
    # #     x, y, r = map(int, circles_pupil)
    # #     # outer circle
    # #     cv.circle(img, (x, y), r, (0, 255, 0), 2)
    # #     # center point
    # #     cv.circle(img, (x, y), 2, (0, 0, 255), -1)
    # # else:
    # #     print("No pupil circles detected")

    # # cv.imwrite("dataset/1/1_pupil.jpg", img)

    # b,g,r = cv.split(img)

    # layer = r
    # # low, high = np.percentile(layer, (1, 99))
    
    # # layer = np.clip((layer.astype(np.float32) - low) / (high - low), 0, 1)
    # # layer = (layer * 255).astype(np.uint8)

    # low, high = np.percentile(layer, (1, 99))

    # # Normalize and clip to [0, 1]
    # gray_norm = np.clip(
    #     (layer.astype(np.float32) - low) / (high - low),
    #     0,
    #     1
    # )

    # # Apply gamma = 0.5
    # grayp = np.power(gray_norm, 0.5)

    # # Convert back to uint8 [0, 255]
    # layer = (grayp * 255).astype(np.uint8)

    # g_blurred7 = cv.GaussianBlur(layer,(7,7),1)
    # g_blurred14 = cv.GaussianBlur(layer,(15,15),2)
    # g_blurred21 = cv.GaussianBlur(layer,(21,21),3)
    # g_blurred27 = cv.GaussianBlur(layer,(27,27),4)
    # g_blurred35 = cv.GaussianBlur(layer,(35,35),5)
    # g_blurred43 = cv.GaussianBlur(layer,(43,43),7)
    # g_blurred51 = cv.GaussianBlur(layer,(51,51),8)

    # # # Canny Detection Test
    # cannyT = 25
    # edges7 = cv.Canny(g_blurred7,cannyT/2,cannyT)
    # edges14 = cv.Canny(g_blurred14,cannyT/2,cannyT)
    # edges21 = cv.Canny(g_blurred21,cannyT/2,cannyT)
    # edges27 = cv.Canny(g_blurred27,cannyT/2,cannyT)
    # edges35 = cv.Canny(g_blurred35,cannyT/2,cannyT)
    # edges51 = cv.Canny(g_blurred51,cannyT/2,cannyT)

    # plt.subplot(221),plt.imshow(img)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(222),plt.imshow(edges7,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(223),plt.imshow(edges14,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224),plt.imshow(edges21,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.tight_layout()
    # plt.show()
    # plt.close("all")
    
    # Circle detection test
    # circles = cv.HoughCircles(g_blurred35,cv.HOUGH_GRADIENT,1,50,
    #                         param1=cannyT,param2=30,minRadius=90,maxRadius=150)

    # # circles = cv.HoughCircles(g_blurred,cv.HOUGH_GRADIENT_ALT,1,5,
    # #                         param1=54,param2=0.8,minRadius=0,maxRadius=0)
    
    # cimg = cv.cvtColor(g,cv.COLOR_GRAY2BGR)
    
    # circles = np.uint16(np.around(circles))
    # for i in circles[0,:]:
    #     print(i)
    #     # draw the outer circle
    #     cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    # circle = detect_iris(g_blurred35)
    # if circles is not None:
    #     print(circles)
    #     x, y, r = map(int, circles)
    #     # outer circle
    #     cv.circle(img, (x, y), r, (0, 255, 0), 2)
    #     # center point
    #     cv.circle(img, (x, y), 2, (0, 0, 255), -1)
    #     # print("No pupil circles detected")

    # cv.imshow("Green Layer", img)

    # cv.waitKey(0)
    # cv.destroyAllWindows()


