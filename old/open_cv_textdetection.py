import numpy as np
import cv2
from PIL import Image
import pytesseract

cap = cv2.VideoCapture(0)

fpsOriginal = cap.get(cv2.CAP_PROP_FPS)
print("original : {0}".format(fpsOriginal))
def doSomething(cap):
    while(1):
        try:

            # Take each frame
            _, img= cap.read()
            # img= cap.grab()

            # for visualization
            vis = img.copy()

            # Extract channels to be processed individually
            channels = cv2.text.computeNMChannels(img)
            # Append negative channels to detect ER- (bright regions over dark background)
            cn = len(channels) - 1

            for c in range(0, cn):
                channels.append((255 - channels[c]))

            # Apply the default cascade classifier to each independent channel (could be done in parallel)
            # print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
            # print("    (...) this may take a while (...)")

            # print len(channels)
            for i,channel in enumerate(channels):
                # if(i>3):
                #     continue;
                erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
                er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

                erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
                er2 = cv2.text.createERFilterNM2(erc2, 0.5)

                regions = cv2.text.detectRegions(channel, er1, er2)

                rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
                # rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

                # Visualization
                for r in range(0, np.shape(rects)[0]):
                    rect = rects[r]

                    imCrop = vis[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]
                    # imCrop = vis[int(rect[1]):int(rect[3] - rect[1]), int(rect[0]):int(rect[2] - rect[0] )]
                    # cv2.imshow('imCrop', imCrop)
                    tesImg = Image.fromarray(imCrop)
                    txt = pytesseract.image_to_string(tesImg)
                    print("found text:"+ txt)
                    cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,0,255), 1)

                    cv2.putText(vis, txt, (rect[0],rect[1]+rect[3]), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 2, 2)

                    # cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)
            # Visualization

            cv2.imshow("Text detection result", vis)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        except:
            continue

    cv2.destroyAllWindows()
doSomething(cap)
