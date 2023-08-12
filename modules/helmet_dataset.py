from bs4 import BeautifulSoup
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations
import albumentations.pytorch


################################################################
# 라벨링이 <태그> 형식으로 되어있는 경우 데이터 전처리 메소드들  
#-------------------------------------------------------------------------------
def generate_box(obj):
    '''
    annotations.xml에서 바운딩 박스 좌표들을 가져오는 함수 
    [parameters]
        obj : annotations.xml 파일에서 읽은 object 
    [return]
        [xmin, ymin, xmax, ymax]

    '''

    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)

    # 바운딩 박스 좌표 (xmin, ymin, xmax, ymax)를 배열로 반환
    return [xmin, ymin, xmax, ymax]  


#-------------------------------------------------------------------------------
def generate_label(obj):
    '''
    annotations.xml에서 바운딩 박스 라벨들을 반환하는 함수
    [parameters]
        obj : annotations.xml 파일에서 읽은 object 
    [return]
        helmet인 경우 return 1
        head인 경우 return 2
        그 외 return 0
    '''
    
    if obj.find('name').text == "helmet": return 1  # 라벨이 helmet 인 경우 return 1
    elif obj.find('name').text == "head": return 2  # 라벨이 head 인 경우 return 2

    return 0


#-------------------------------------------------------------------------------
def generate_target(file):
    '''
    generate_box, generate_label 함수를 이용해 받은 box, label 정보를
    target dictionary 형태로 반환하는 함수 
    [parameters]
        file : annotation.xml 파일을 입력
    [return]
        target = {'boxes': [...], 'labels': [...]} 
    '''
    with open(file) as f:
        data = f.read()                            # XML 파일을 읽기
        soup = BeautifulSoup(data, "html.parser")  # BeautifulSoup으로 파싱
        objects = soup.find_all("object")          # "object" 태그를 모두 찾아 객체 목록을 생성

        num_objs = len(objects)

        boxes = []
        labels = []

        # 객체 목록을 반복하여 각 객체에 대한 바운딩 박스와 레이블을 생성
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        # 반환할 딕셔너리(target)에 boxes와 labels 리스트를 추가
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # 생성한 딕셔너리(target) 반환
        return target  


#-------------------------------------------------------------------------------
# 배치의 이미지 및 어노테이션을 모으는 함수 
# 데이터 로더에 의해 호출되어 배치를 수집하는 동안 사용될 예정인 함수
# 데이터 로더를 사용하여 커스텀 데이터셋에서 배치를 생성하고 싶을 때 사용됨

# def collate_fn(batch):
#     return tuple(zip(*batch))\
    
def collate_fn(batch):
    images = []
    annots = []

    for sample in batch:
        image, annot = sample
        images.append(image)
        annots.append(annot)

    return images, annots



###################################################################
# 데이터셋 클래스 생성 
class AlbumentationsDataset(Dataset):
    '''
    [parameters]
        path : image 파일 경로
        ann_path : annotations 파일 경로
        transform : 데이터 전처리용 transformer
    [method]
        len : images 개수 반환
        getitem : index를 받아 해당 이미지 및 annotation 반환

    '''
    def __init__(self, path, ann_path, transform=None):
        self.path = path
        self.ann_path = ann_path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.annos = list(sorted(os.listdir(self.ann_path)))
        self.transform = transform
            # 선택적으로 Albumentations 라이브러리를 사용한 전처리 변환 함수

    def __len__(self):
        return len(self.imgs)

    # 주어진 인덱스의 이미지 및 해당하는 어노테이션을 반환하는 메소드 #
    def __getitem__(self, idx):
        file_image = self.imgs[idx]
        file_label = self.annos[idx]
        img_path = os.path.join(self.path, file_image)
        anno_path = os.path.join(self.ann_path, file_label)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = generate_target(anno_path)

        # Albumentations 변환 함수가 입력되면 -> image와 target에 transform 적용
        if self.transform:
            transformed = self.transform(image=image, bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            target = {'boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32), 'labels': torch.tensor(transformed['labels'])}

        return image, target
    




###################################################################
# Transformer for Dataset 
bbox_transform = albumentations.Compose([
    albumentations.Resize(224, 224),  # size 300 -> 224로 변경 <- new!
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화 
    albumentations.pytorch.transforms.ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)],
    bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']),
)