import cv2

cap = cv2.VideoCapture(0)
fpsOriginal = cap.get(cv2.CAP_PROP_FPS)
print("original : {0}".format(fpsOriginal))

while(1):

    # Take each frame
    _, img = cap.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('asdfasdf', grey)