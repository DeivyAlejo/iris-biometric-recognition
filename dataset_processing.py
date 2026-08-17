import cv2 as cv
import os
import glob
import numpy as np
import preprocessing_image

for i in range(20):
    dir = f"dataset/{i+1}/*.tiff"
    # print(dir)

    path = glob.glob(dir)
    for img in path:
        base_name = os.path.splitext(os.path.basename(img))[0]
        image = cv.imread(img)
        # b,g,r = cv.split(image)

        circle = preprocessing_image.detect_iris(image)

        if circle is not None:
            x, y, r = map(int, circle)
            h, w = image.shape[:2]

            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, w)
            y2 = min(y + r, h)

            cropped = image[y1:y2, x1:x2].copy()
            if cropped.size == 0:
                print(f"Invalid crop for {img}, saving original image.")
                output = image
            else:
                mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
                cx = x - x1
                cy = y - y1
                cv.circle(mask, (cx, cy), r, 255, -1)
                output = cv.bitwise_and(cropped, cropped, mask=mask)

        else:
            print(f"No iris detected in {img}")
            output = image

        cv.imwrite(f"dataset/{i+1}/{base_name}.jpg", output)
        # cv.imshow(img, image)

        # cv.waitKey(0)
        # cv.destroyAllWindows()

cv.destroyAllWindows()