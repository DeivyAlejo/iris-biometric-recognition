import cv2 as cv
import os
import glob
import preprocessing_image

for i in range(20):
    dir = f"dataset/{i+1}/D/*.tiff"
    print(dir)

    path = glob.glob(dir)
    for img in path:
        base_name = os.path.splitext(os.path.basename(img))[0]
        image = cv.imread(img)
        # b,g,r = cv.split(image)

        circle = preprocessing_image.detect_iris(image)
        if circle is not None:
            print(circle)
            # draw the outer circle
            cv.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(image,(circle[0],circle[1]),2,(0,0,255),3)

        cv.imwrite(f"dataset/{i+1}/D/{base_name}.jpg", image)
        # cv.imshow(img, image)

        # cv.waitKey(0)
        # cv.destroyAllWindows()

cv.destroyAllWindows()