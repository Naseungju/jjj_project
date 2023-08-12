import urllib.request
import cv2
import numpy as np
import time
from easyocr import Reader
# from modules import *

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://172.30.1.60:8080/shot.jpg'

# OCR Reader
reader = Reader(lang_list=['en'])

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    
    # 이미지 사이즈 줄이기
    img = cv2.resize(img, (720, 486), interpolation=cv2.INTER_AREA)

    ###################################
    # # 이미지 전처리 (옵션)
    # processed_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    # processed_frame = cv2.medianBlur(processed_frame, ksize=3)  # noise 제거
    # processed_frame = change_contrast(processed_frame, 1)  # 대비 높임

    # OCR 실행
    s_time = time.time()
    results = reader.readtext(img, detail = 0)
    e_time = time.time()
    print(e_time - s_time)

    if results:
        # 인식된 텍스트를 화면에 출력
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_with_text = cv2.putText(img, results[0], (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # put the image on screen
        cv2.imshow('IPWebcam', frame_with_text)
    else:
        cv2.imshow('IPWebcam', img)

    # To give the processor some less stress
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
