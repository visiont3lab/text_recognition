import cv2
import numpy as np
from detection import DETECTION
from transform import four_point_transform
import glob
import pytesseract
from pytesseract import Output
import re
import time

def rotate(image, center=None, scale=1.0):
    angle = 360 - int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def tess_ocr(image, use_threshold = False):
    img = rotate(image)
    if use_threshold:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 12 --oem 1')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = str(d['text'][i])
        if text:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img

# class construct
mydetection = DETECTION()
mydetection.refiner = False
mydetection.load_model()

images = [cv2.imread(file) for file in glob.glob("edited/*.jpg")]
idx = 0
kernel = np.ones((3,3),np.uint8)
clahe = cv2.createCLAHE()

for image_opencv in images:
    try:
        start = time.process_time()
        # INFERENCE #
        bboxes, polys = mydetection.test_net(image_opencv)
        w = image_opencv.shape[0] # + image_opencv.shape[1]
        h = image_opencv.shape[1]
        out = np.ones((w, h), np.uint8) * 255
        # old_points = []
        ## SQUARE BOXES
        for i in range(len(bboxes)):
            points = []
            for j in range(len(bboxes[i])):
                a = (np.rint(bboxes[i][j][0]), np.rint(bboxes[i][j][1]))
                if (j == (len(bboxes[i]) - 1)):
                    b = (np.rint(bboxes[i][0][0]), np.rint(bboxes[i][0][1]))
                else:
                    b = (np.rint(bboxes[i][j + 1][0]), np.rint(bboxes[i][j + 1][1]))
                cv2.line(image_opencv, a, b, (255, 255, 255), 1)
                points.append(a)

            points = np.array(points, dtype='int32')
            roi = four_point_transform(image_opencv, points)
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #img_dil = cv2.erode(img, kernel, iterations=1)
            #img = cv2.dilate(img_dil, kernel, iterations=1)
            #img = img_dil
            # for k in enumerate(img):
            start_p_x = points[0][1]
            start_p_y = points[0][0]
            # if i > 0:
            #     old_points_x = points[0][1]
            #     old_points_y = points[0][0]
            #     if old_points_x > start_p_x and old_points_x < start_p_x + img.shape[0]:
            #         start_p_x = old_points_x
            #     if old_points_y > start_p_y and old_points_y < start_p_y + img.shape[1]:
            #         start_p_y = old_points_y
            out[start_p_x:start_p_x + img.shape[0], start_p_y:start_p_y + img.shape[1]] = img
            # old_points = points
        # cv2.imwrite('./zfolder/test' + str(i) + ".jpg", img)
        # save = cv2.imwrite('./zfolder/test.jpg',img)
        print("detect. img", idx, ":", time.process_time() - start)
        cv2.imwrite('./zfolder/test_comp' + str(idx) + ".jpg", out)

        result = tess_ocr(out)
        print("total img", idx, ":", time.process_time() - start)
        cv2.imwrite('./zfolder/result ' + str(idx) + ".jpg", result)
        idx = idx + 1
        # IMSHOW
        #cv2.imshow("Text detection", image_opencv)
    except:
        idx = idx + 1
        continue
