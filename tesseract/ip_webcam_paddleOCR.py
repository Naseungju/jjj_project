import urllib.request
import cv2
import numpy as np
import time
from paddleocr import PaddleOCR
# from modules import change_contrast


url='http://192.168.0.174:8080/shot.jpg'
ocr = PaddleOCR()


while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)  # 뭐여
    
    # 이미지 사이즈 줄이기
    img = cv2.resize(img, (720, 486), interpolation=cv2.INTER_AREA)

    ######################################
    # OCR 실행
    result = ocr.ocr(img)

    # 인식된 텍스트를 화면에 출력
    if result:
        for res in result[0]:
            text = res[1][0]  # text
            percentage = res[1][1]  # 확률
            show_text = text + ' : ' + str(round(percentage,3))

            # 인식된 텍스트를 화면에 출력
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame_with_text = cv2.putText(img, show_text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # put the image on screen
            cv2.imshow('IPWebcam', frame_with_text)
    else:
        cv2.imshow('IPWebcam', img)

    # To give the processor some less stress
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
