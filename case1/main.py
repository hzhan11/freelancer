import glob

import cv2
import numpy as np
import pytesseract
from PIL import Image

def filter_out_red(src_frame):
    if src_frame is not None:
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([175, 175, 175])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return cv2.bitwise_and(src_frame, src_frame, mask=mask)

def fillHole(im_in):
    im_floodfill = im_in.copy()
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv
    return im_out

def fileout(img):
    org = img.copy()
    org = cv2.cvtColor(org, cv2.COLOR_RGBA2GRAY)
    img = filter_out_red(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret1, th1 = cv2.threshold(gray, 12, 250, cv2.THRESH_BINARY)
    img = fillHole(th1)
    img = cv2.bitwise_and(img, org)
    return img

def match(img_rgb, template):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        right_bottom = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
    return img_rgb, zip(*loc[::-1])

def remove_duplicate(pts1, pts2, w):
    pp = list(pts1)
    pp.extend(list(pts2))
    newlist = []
    for p in pp:
        far = True
        for pp in newlist:
            if np.linalg.norm(np.array(p) - np.array(pp)) < w*3/4:
                far = False
                break
        if far:
            newlist.append(p)
    return newlist

def text_preprocess(gray):

    color_i = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = cv2.imread("new/t1.png")
    mask_inv = cv2.bitwise_not(mask)
    color_i = cv2.bitwise_or(color_i, mask_inv)
    gray = cv2.cvtColor(color_i, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (80, 80))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    gray = cv2.filter2D(gray, -1, kernel=kernel)
    ret, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return gray


def gen_string_list(img, pts, w):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count = 0
    string_img_list = []
    offset = 0
    for pp in pts:
        bnd = img_gray[int(pp[1])+offset:int(pp[1])+w-offset,int(pp[0])+offset:int(pp[0])+w-offset]
        bnd = text_preprocess(bnd)
        cv2.imwrite("new/bnd/%d.png"%count, bnd)
        string_img_list.append(bnd)
        count += 1
    return string_img_list


def ocr(file):
    config = "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    file = cv2.cvtColor(file, cv2.COLOR_GRAY2RGB)
    text = pytesseract.image_to_string(file, lang="Penitentiary", config=config)
    text = text.replace("\n", ",")

    text_img = np.zeros((file.shape[0], 500, 3), np.uint8)
    text_img = cv2.putText(text_img, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
    text_img = cv2.hconcat([file, text_img])
    return text, text_img

import sys

if __name__ == '__main__':
    
    w = 42

    input = sys.argv[1]
    source_img = cv2.imread(input)
    source_img = cv2.resize(source_img, (2822,1995))
    s1 = fileout(source_img)
    s1 = cv2.cvtColor(s1, cv2.COLOR_GRAY2BGRA)
    cv2.imwrite("new/step1.png", s1)

    template1 = cv2.imread('new/p1.png', 0)
    s1, pts1 = match(s1, template1)
    template2 = cv2.imread('new/p2.png', 0)
    s2, pts2 = match(s1, template2)
    cv2.imwrite("new/step2.png", s2)
    
    pts = remove_duplicate(pts1, pts2, w)
    print("total %d tags found as below"%len(pts))

    img_list = gen_string_list(source_img, pts, w)
    #s = None
    for file in img_list:
        text, _ = ocr(file)
        print(text)
    #    if s is None:
    #        s = img
    #    else:
    #        s = cv2.vconcat([s, img])

    #cv2.imshow("haha",s)
    #cv2.waitKey()
