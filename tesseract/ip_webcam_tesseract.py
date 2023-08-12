import urllib.request
import cv2
import numpy as np
import time
import pytesseract
import re
from modules import change_contrast

url='http://172.30.1.60:8080/shot.jpg'


while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1)
    
    # 이미지 사이즈 줄이기
    img = cv2.resize(img, (720, 486), interpolation=cv2.INTER_AREA)

    ######################################
    # 이미지 전처리 (옵션)
    processed_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    processed_frame = cv2.medianBlur(processed_frame, ksize=3)  # noise 제거
    processed_frame = change_contrast(processed_frame, 1)  # 대비 높임

    ######################################
    # OCR 실행
    config = '--psm 6 outputbase digits'  # 숫자만 찾는 모드
    ocr_result = pytesseract.image_to_string(img, config=config)
    
    # 영역 추출
    ocr_result = re.findall("\d+", ocr_result)
    if ocr_result:
        for num in ocr_result:
            if len(num) == 10: text='worker id: ' + num  # 읽은 숫자의 길이가 4일 경우에만 출력
            else: text = 'no read'
    else: text = 'no read'

    # 인식된 텍스트를 화면에 출력
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_with_text = cv2.putText(img, text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)   
    
    # 화면에 출력
    cv2.imshow('frame', frame_with_text)

    # To give the processor some less stress
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
