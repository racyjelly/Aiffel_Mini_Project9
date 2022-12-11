import os
import urllib
import cv2
import numpy as np
import pixellib
import inspect
import tensorflow as tf
print(inspect.getfile(pixellib))
print(tf.__version__)
from pixellib.semantic import semantic_segmentation
from matplotlib import pyplot as plt

image_path = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject9\images\jennie.jpg"
img_ori = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
plt.show()

model_file = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject9\models\deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"

# PixelLib가 제공하는 모델의 url
model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5' 


urllib.request.urlretrieve(model_url, model_file) 
# urllib 패키지 내에 있는 request 모듈의 urlretrieve 함수를 이용해서 model_url에 있는 파일을 다운로드 해서
# model_file 파일명으로 저장

model = semantic_segmentation()
model.load_pascalvoc_model(model_file)
# error 해결: https://stackoverflow.com/questions/73084941/valueerror-you-are-trying-to-load-a-weight-file-containing-293-layers-into-a-mo


segvalues, output = model.segmentAsPascalvoc(image_path) 
# segmentAsPascalvoc()함 수 를 호출 하여 입력된 이미지를 분할, 분할 출력의 배열을 가져옴
# 분할 은 pacalvoc 데이터로 학습된 모델을 이용

LABEL_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
]
len(LABEL_NAMES)

plt.imshow(output)
plt.show()

print(segvalues)
# segmentAsPascalvoc() 함수를 호출하여 입력된 이미지를 분할한 뒤
# 나온 결과값 중 배열값을 출력

#segvalues에 있는 class_ids를 담겨있는 값을 통해 pacalvoc에 담겨있는 라벨을 출력
for class_id in segvalues['class_ids']:
    print(LABEL_NAMES[class_id])
# segvalues에는 class_ids와 masks가 있음
# class_ids를 통해 어떤 물체가 담겨 있는지 알수 있음

#컬러맵 만들기 
colormap = np.zeros((256, 3), dtype = int)
ind = np.arange(256, dtype=int)

for shift in reversed(range(8)):
    for channel in range(3):
        colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

colormap[:20] #생성한 20개의 컬러맵 출력

print(colormap[15])
# colormap의 output이미지가 BGR 순서로 배치 -> RGB로 바꿔줘야함
seg_color = (colormap[15][1], colormap[15][-1], colormap[15][0])
seg_map = np.all(output==seg_color, axis=-1)
print(seg_map.shape)
plt.imshow(seg_map, cmap='gray')
plt.show()

# 원본이미지를 img_show에 할당한 뒤
# 이미지 사람이 있는 위치와 배경을 분리해서 표현한 color_mask 를 만든 후
# 두 이미지를 합쳐서 출력
img_show = img_ori.copy()

# True과 False인 값을 각각 255과 0으로 바꿔줍니다
img_mask = seg_map.astype(np.uint8) * 255

# 255와 0을 적당한 색상으로 바꿔봅니다
color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)

# 원본 이미지와 마스트를 적당히 합쳐봅니다
# 0.6과 0.4는 두 이미지를 섞는 비율입니다.
img_show = cv2.addWeighted(img_show, 0.6, color_mask, 0.4, 0.0)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()

bg_dir = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject9\images\caffe4.jpg"
img_bg = cv2.imread(bg_dir)
img_bg = cv2.resize(img_bg, (img_ori.shape[1], img_ori.shape[0]), interpolation=cv2.INTER_CUBIC)
img_ori_blur = cv2.blur(img_bg, (10, 10))
# img_ori_blur = cv2.blur(img_ori, (13, 13))
# (13, 13) blurring kernel size
plt.imshow(cv2.cvtColor(img_ori_blur, cv2.COLOR_BGR2RGB))
plt.show()

img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
# cv2.bitwise_not(): 이미지가 반전됩니다. 배경이 0 사람이 255 였으나
# 연산을 하고 나면 배경은 255 사람은 0입니다.
img_bg_mask = cv2.bitwise_not(img_mask_color)
plt.imshow(cv2.cvtColor(img_bg_mask, cv2.COLOR_BGR2RGB))
plt.show()
# cv2.bitwise_and()을 사용하면 배경만 있는 영상을 얻을 수 있습니다.
# 0과 어떤 수를 bitwise_and 연산을 해도 0이 되기 때문에 
# 사람이 0인 경우에는 사람이 있던 모든 픽셀이 0이 됩니다.
# 결국 사람이 사라지고 배경만 남게 됨
img_bg_blur = cv2.bitwise_and(img_ori_blur, img_bg_mask)
plt.imshow(cv2.cvtColor(img_bg_blur, cv2.COLOR_BGR2RGB))
plt.show()

# Conditioning Darkness
# https://velog.io/@oosiz/OpenCV-Python-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B0%9D%EA%B8%B0-%EC%A1%B0%EC%A0%95
val = 30
dark_array = np.full(img_ori.shape, (val, val, val), dtype=np.uint8)
sub_dst = cv2.subtract(img_ori, dark_array)
sub_dst = cv2.bilateralFilter(sub_dst, -1, 40, 10)
# https://wjh2307.tistory.com/11

img_concat = np.where(img_mask_color==255, sub_dst, img_bg_blur)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
# https://numpy.org/doc/stable/reference/generated/numpy.where.html
print("End to change Background")



