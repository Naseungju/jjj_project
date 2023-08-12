import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


###################################################
#-------------------- 시각화 ---------------------#
def plot_image_from_output(img, annotation, ax):
    '''
    모델 통과 후 출력값으로 bounding box를 그린 이미지 출력 메소드
    [parameters]
        img : 모델 반환값 이미지
        annotation : 모델 반환값 bounding box 좌표 정보
        ax : 
    [return] 
    '''
    img = img.permute(1,2,0)  # 이미지를 다시 원래 형태로 되돌림 
    img = img.cpu().numpy()   # numpy 객체 처리를 위해 cpu로 이동 
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx].cpu().numpy()

        if annotation['labels'][idx] == 0 :  # none
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        elif annotation['labels'][idx] == 1 :  # helmet
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')

        else :  # head
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)




###################################################
#--------------- 조기 종료 클래스 ----------------#
class EarlyStopping():
    def __init__(self, patience=5, save_path=None, target_loss=100, model_name='retina'):
        # 초기화
        self.best_loss = 0
        self.patience_count = 0
        self.target_loss = target_loss
        self.patience = patience
        self.save_path = save_path
        best_model_name = model_name + '_best.pth'
        self.best_model_path = self.save_path + best_model_name
        last_model_name = model_name + '_last.pth'
        self.last_model_path = self.save_path + last_model_name

    # 얼리 스토핑 여부 확인 함수 정의
    def is_stop(self, model, loss):
        # 모델 저장(마지막 모델)
        self.__save_last_model(model)
        # 베스트 스코어가 타겟 스코어보다 낮을 경우 (베스트 손실이 타겟 로스보다 낮을경우)
        if self.best_loss > self.target_loss:
            # 스코어가 이전보다 안좋을 경우 (손실이 이전보다 더 높을경우)
            if self.best_loss <= loss:
                # patience 초기화
                self.patience_count = 0
                return False
            # 스코어를 업데이트
            self.best_loss = loss
            # 모델 저장
            self.__save_best_model(model)
            # patience 초기화
            self.patience_count = 0
            return False
        # 스코어가 이전보다 좋을 경우
        if self.best_loss < loss:
            # 스코어를 업데이트
            self.best_loss = loss
            # 모델 저장
            self.__save_best_model(model)
            # patience 초기화
            self.patience_count = 0
            return False
        # 스코어가 이전보다 좋지 않을 경우 +
        # 스코어가 타겟 스코어보다 높을 경우
        # patience 증가
        self.patience_count += 1
        # patience가 최대치를 넘을 경우
        if self.patience_count > self.patience:
            return True
        # patience가 최대치를 넘지 않을 경우
        return False

    # 모델 저장 함수 정의
    def __save_best_model(self, model):
        torch.save(model.state_dict(), self.best_model_path)

    # 마지막 모델 저장 함수 정의
    def __save_last_model(self, model):
        torch.save(model.state_dict(), self.last_model_path)




###################################################
# ---------------- 예측 메소드 ------------------ #
def make_prediction(model, img, threshold):
    '''
    [parameters]
        model : 모델 
        img : 추론할 이미지
        threshold : bounding box로 판별할 임계치
    [return]
        preds : 예측값 
    '''
    model.eval()
    preds = model(img)

    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :      # threshold 넘는 idx 구함
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds