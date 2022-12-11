"""
webcam이 있는 퍼실님들이 노드 검사를 해야함
frame 끊김이 있으니 주의하기

배경전환 크로마키 사진은  change_background.py 프로젝트에서 확인 가능

인물 사진에서의 발생하는 문제점은 segmentation한 인물과 배경이 조화롭지 않음
이는 배경이 blur처리가 되어 있으므로 인물도 blur 처리를 해주어야 함
그러나 어떻게 해주느냐가 관건

** 그래서 blur처리를 cv2.bilateralFilter 를 원본이미지에 주어서
blur처리된 배경과 어울러지게 처리함 (change_background.py)
양방향 필터의 경우 에지가 아닌 부분에서 blurring처리를 하기 때문에
물체의 윤곽이 잘 보존됨

webcam과 semantic segmentation model을 연결한 이유는
추후에 아이펠톤에서의 활용 가능성을 생각하여 연동함

Frame 끊김 해결 방안: GPU를 쓰면 Frame끊김이 덜해진다.
"""

import os
import urllib
import cv2
import numpy as np
print("numpy version: ", np.version.version)
import pixellib
import inspect
import tensorflow as tf
print(inspect.getfile(pixellib))
print("tensorflow version: ", tf.__version__)
from pixellib.semantic import semantic_segmentation

# Settings (Labelnames, Colormap)
Label_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']
colormap = np.zeros((256, 3), dtype=int)


def start_seg_model(model_url, model_weight):

    # segmentation model load weights
    urllib.request.urlretrieve(model_url, model_weight)
    model = semantic_segmentation()
    model.load_pascalvoc_model(model_weight)
    "pip install --user h5py"

    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel)&1) << shift
        ind >>=3
    
    seg_color = (colormap[15][1], colormap[15][-1], colormap[15][0])

    cab = cv2.VideoCapture(0)
    webcam = cab.isOpened()
    if webcam==False:
        print("Could not open Webcam")
        exit()
    while webcam==True:
        ret, img = cab.read()

        if ret:
            cv2.imwrite("Frame.jpg", img)
            filename = "Frame.jpg"

            segvalues, output = model.segmentAsPascalvoc(filename)
            img_show = img.copy()
            seg_map = np.all(output==seg_color, axis=-1)
            img_mask = seg_map.astype(np.uint8)*255
            img_ori_blur = cv2.blur(img_show, (13, 13))
            img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
            img_bg_mask = cv2.bitwise_not(img_mask_color)
            img_bg_blur = cv2.bitwise_and(img_ori_blur, img_bg_mask)
            img_concat = np.where(img_mask_color==255, img, img_bg_blur)

            cv2.imshow("Video", img_concat)
        
        if cv2.waitKey(1) & 0xFF ==ord("q"): # 단축키 q 눌러서 중지하기
            break
    
    cab.release()
    cv2.destroyAllWindows()

if __name__=="__main__":

    model_weight = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject9\models\deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
    start_seg_model(model_url=model_url, model_weight=model_weight)



    


