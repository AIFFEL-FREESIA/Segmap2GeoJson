import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import click
import numpy as np
import tensorflow as tf
import random
import math

import segmentation_models as sm
import albumentations as A


def init_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            click.echo(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            click.echo("\033[01m\033[33mGPU Found 😇\033[0m")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            click.echo(e)
            click.echo("\033[01m\033[33mNo GPU. Work will be slow😅\033[0m")

init_gpu()


#########################################################
#                        모델 정의                        #
#########################################################
BACKBONE = 'efficientnetb3'
CLASSES = ['building', 'road']
MODEL_PATH = './model/E15_4BnR.h5'
COLOR_MAP = [
    (165, 42, 42),
    (0, 192, 0),
    (255,255,255)
]


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
click.echo("\033[01m\033[33mLoad Segementation model ... ⏳\033[0m")
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation, encoder_weights='imagenet')
model.load_weights(MODEL_PATH)
click.echo("\033[01m\033[33mNow the model is ready ✊\033[0m")


#########################################################
#                  Inference 함수 정의                    #
#########################################################

def get_slice_pos(split_cnt, slice_type='sliding', stride_size=192, input_size=256, img_size=1024):
    pts = []
    
    if slice_type == 'sliding':
        n = math.sqrt(split_cnt)
        for slice_pos in range(split_cnt):
            pos_i = int(math.floor(slice_pos / n))
            pos_j = int(slice_pos % int(n))
            
            x = stride_size * pos_i
            y = stride_size * pos_j
            
            pts.append((x, x+input_size, y, y+input_size))
            
    elif slice_type =='crop':
        random.seed(2)
        get_random_point = lambda: random.randint(0, img_size-input_size)
        
        for _ in range(split_cnt):
            x = get_random_point()
            y = get_random_point()
            pts.append((x, x+input_size, y, y+input_size))
            
    return pts

def merge_img(sub_imgs, pts):
    merged = np.zeros((1024, 1024, 3))        # sub_imgs를 합친 이미지
    seg_map = np.zeros((1024, 1024))          # segmentation map, (0, 1, 2) 로 구성 ; (3, 1024, 1024)
    
    for i, (x0, x1, y0, y1) in enumerate(pts):
        sub_a = merged[x0:x1, y0:y1, :]        # [[0,0,0], [0,0,0], ...]
        sub_b = sub_imgs[i]
        added = np.where(sub_a > sub_b, sub_a, sub_b)
        reduced = np.argmax(added, axis=-1)   # 0, 1, 2 channel 중 큰 것
        merged[x0:x1, y0:y1, :] = added
        seg_map[x0:x1, y0:y1] = reduced
    
    return seg_map

def crop_png(img, pts):
    if len(img.shape) == 3:
        sub_imgs = [img[x0:x1, y0:y1, :] for x0, x1, y0, y1 in pts]
    elif len(img.shape) == 2:
        sub_imgs = [img[x0:x1, y0:y1] for x0, x1, y0, y1 in pts]
    return sub_imgs

def coloring_seg_map(seg_map): # (1024, 1024) segmentation map -> (1024, 1024, 3)
    color_map = np.stack([seg_map, seg_map, seg_map], axis=-1)
    
    color_map = np.where(color_map == 0, COLOR_MAP[0], color_map)
    color_map = np.where(color_map == 1, COLOR_MAP[1], color_map)
    color_map = np.where(color_map == 2, COLOR_MAP[2], color_map).astype(np.uint8)
        
    return color_map

def inference(img):
    click.echo('\033[01m\033[33mInference Started ... ⏳\033[0m')
    pts = get_slice_pos(25)
    preprocessing = A.Compose([
        A.Lambda(image=sm.get_preprocessing(BACKBONE)),
    ])
    
    sub_imgs = crop_png(img, pts)
    sub_imgs = [preprocessing(image=img)['image'] for img in sub_imgs]
    sub_imgs = [np.expand_dims(img, axis=0) for img in sub_imgs]
    predict_imgs = [model.predict(img) for img in sub_imgs]
    predict_imgs = [predict.squeeze() for predict in predict_imgs]
    merged_img = merge_img(predict_imgs, pts)
    seg_map = coloring_seg_map(merged_img)
    
    click.echo('\033[01m\033[33mInference done ... ✊\033[0m')
    return seg_map