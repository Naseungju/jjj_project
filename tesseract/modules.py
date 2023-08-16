import cv2
import matplotlib.pyplot as plt
import numpy as np

#################################################################################
#### plt_imshow ####
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    '''
    title - 이미지 제목. default=image.여러 이미지 표시 경우 제목 목록 전달 가능
    img - 이미지 또는 이미지 목록. 단일 이미지 또는 여러 이미지 처리 가능
    figsize - 이미지 표시 크기를 결정하는 숫자 쌍. default=(8,5)
    '''
    plt.figure(figsize=figsize)

    ## img가 이미지 목록인 경우
    if type(img) == list:
        if type(title) == list: titles = title  # title 목록 지정
        else: # title 목록 생성
            titles = []
            for i in range(len(img)):
                titles.append(title)
                
        # 모든 이미지 rgb 변환
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            
        plt.show()
    
    ## img가 단일 이미지인 경우
    else:  # rgb 변환
        if len(img.shape) < 3: rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


###############################################################
#### 대비 조절 함수 ####
def change_contrast(img, alpha=0.0):
    '''
    명암비(contrast)를 변환하는 함수
    [parameter]
        img: ndarray - 명암비를 조절할 대상 이미지
        alpha: float -  대비 조절 비율값. 기본: 0.0(원본) - 양수: 명암비를 높인다. 음수 - 명암비를 낮춘다. 
    [return]
        ndarray - 명암비가 변환된 이미지
    '''
    return np.clip((1.0 + alpha) * img - 128 * alpha, 0, 255).astype('uint8')