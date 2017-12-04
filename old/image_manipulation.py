import cv2
import numpy as np
from PIL import Image
import pytesseract



def workingCanny(cap):
    while (1):
        # Take each frame
        _, frame = cap.read()
        kernel = np.ones((5, 5), np.float32) / 25
        # frame = cv2.medianBlur(frame,1)
        frameBlur = cv2.filter2D(frame, -1, kernel)
        greyBlur = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2GRAY)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 100, 200)
        edges = cv2.Canny(grey, 50, 150, apertureSize=3)
        minLineLength = 100
        maxLineGap = 10
        # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # greyBlur = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        thGaussian = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        laplacianBlur = cv2.Laplacian(greyBlur, cv2.CV_64F)

        # thMean = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # thGaussianBlur = cv2.adaptiveThreshold(greyBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # thMeanBlur = cv2.adaptiveThreshold(greyBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow('frame', frame)
        cv2.imshow('thGaussian', thGaussian)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('canny', canny)
        # cv2.imshow('laplacianBlur',laplacianBlur)
        # cv2.imshow('thGaussianBlur', thGaussianBlur)
        # cv2.imshow('thMeanBlur', thMeanBlur)
        # cv2.imshow('thMean',thMean)
        # cv2.imshow('blur',frameBlur)
        (thresh, bw_img) = cv2.threshold(laplacian, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = Image.fromarray(bw_img)
        txt = pytesseract.image_to_string(img)
        print(txt)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def stackOverflow(cap):
    # https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv
    while (1):

        # Take each frame
        _, img = cap.read()
        # img = cv2.imread(file_name)

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                             3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

        # for cv2.x.x

        # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

        # for cv3.x.x comment above line and uncomment line below

        image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = img_final[y :y +  h , x : x + w]
    
            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1
    
            '''
        # write original image with added contours to disk
        cv2.imshow('captcha_result', img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def stackOverflowWithOCR(cap):
    # https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv
    while (1):

        # Take each frame
        _, img = cap.read()
        # img = cv2.imread(file_name)

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        '''
                line  8 to 12  : Remove noisy portion 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                             3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

        # for cv2.x.x

        # contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

        # for cv3.x.x comment above line and uncomment line below

        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue

            # draw rectangle around contour on original image
            imCrop = dilated[int(y):int(y + h), int(x):int(x + w)]
            # cv2.imshow('imCrop', imCrop)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            tesImg = Image.fromarray(imCrop)
            txt = pytesseract.image_to_string(tesImg)
            print(txt)
            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = img_final[y :y +  h , x : x + w]

            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1

            '''
        # write original image with added contours to disk
        cv2.imshow('captcha_result', img)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def combination(cap):
    while (1):
        # Take each frame
        _, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 100, 200)
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        # edges = cv2.Canny(grey, 50, 150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(canny, kernel, iterations=9)  # dilate , more the iteration more the dilation
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue
            # draw rectangle around contour on original image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), -2)
            # cv2.rectangle(frame, (x,y), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), -2)

            cv2.putText(frame, 'This one', (x, y+h), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, 2)

            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = img_final[y :y +  h , x : x + w]

            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1

            '''
        # write original image with added contours to disk
        cv2.imshow('captcha_result', frame)

        cv2.imshow('canny', canny)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def combinationWithOCR(cap):
    while (1):
        # Take each frame
        _, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(frame, 100, 200)
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        # edges = cv2.Canny(grey, 50, 150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(canny, kernel, iterations=9)  # dilate , more the iteration more the dilation
        image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
            if w < 35 and h < 35:
                continue
            # draw rectangle around contour on original image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            imCrop = dilated[int(y):int(y + h), int(x):int(x + w)]
            # cv2.imshow('imCrop', imCrop)
            tesImg = Image.fromarray(imCrop)
            txt = pytesseract.image_to_string(tesImg)
            print(txt)
            '''
            #you can crop image and send to OCR  , false detected will return no text :)
            
            s = file_name + '/crop_' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            index = index + 1

            '''
        # write original image with added contours to disk
        cv2.imshow('captcha_result', frame)

        cv2.imshow('canny', canny)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def deeptextdetection(cap) :
    while (1):
        # Take each frame
        _, img= cap.read()
        textSpotter = cv2.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
        rects, outProbs = textSpotter.detect(img);
        vis = img.copy()
        thres = 0.6

        for r in range(np.shape(rects)[0]):

            if outProbs[r] > thres:
                rect = rects[r]
                cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

        cv2.imshow("Text detection result", vis)

def grab(cap):
    frame = cv2.VideoCapture.grab()
    # frame = cv2.VideoCapture.retrieve(cap);
    ret, f = cv2.VideoCapture.retrieve(frame)
cap = cv2.VideoCapture(0)
fpsOriginal = cap.get(cv2.CAP_PROP_FPS)
print("original : {0}".format(fpsOriginal))

# deeptextdetection(cap)
# workingCanny(cap)
# stackOverflowWithOCR(cap)
# grab(cap)
# stackOverflow(cap)









combination(cap)
# combinationWithOCR(cap)


