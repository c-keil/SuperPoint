from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import threading
from multiprocessing import Pool
from multiprocessing import cpu_count

coco = COCO('../../data/COCO/annotations/instances_train2014.json')
img_dir = '../../data/COCO/train2014'
target_dir = '../../data/COCO/train2014_syn'
image_ids = coco.getImgIds()


# from adrian
def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]


def create_synthtic(image_id):
    img = coco.imgs[image_id]
    # loading annotations into memory...
    # Done (t=12.70s)
    # creating index...
    # index created!
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
    print(np.floor(255 / len(anns_ids)))
    anns = coco.loadAnns(anns_ids)
    binmask = coco.annToMask(anns[0]) * anns_ids[0] * np.floor(255 / len(anns_ids))
    print(os.path.join(img_dir, img['file_name']))
    originImage = cv2.imread(os.path.join(img_dir, img['file_name']), cv2.IMREAD_GRAYSCALE)
    cv2.imshow("image", originImage)
    synthetic_img = np.zeros_like(originImage, dtype=np.float32)
    synthetic_img[:] = random.randint(-10, 10) * 15
    for i in range(len(anns)):
        binmask += coco.annToMask(anns[i]) * anns_ids[i] * np.floor(255 / len(anns_ids))
        mask = coco.annToMask(anns[i])
        intensity_variation = random.randint(-10, 10) * 15
        synthetic_img[mask > 0] = intensity_variation

    # cv2.imshow("mask before blur", synthetic_img.astype(np.uint8))
    synthetic_img = cv2.GaussianBlur(synthetic_img, (11, 11), 0)
    mask_bblur = synthetic_img
    # cv2.imshow("mask after blur", synthetic_img)
    # cv2.waitKey(0)
    print(originImage.dtype)
    synthetic_img = originImage.astype(np.float32) + synthetic_img
    cv2.normalize(synthetic_img, synthetic_img, 0, 1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(mask_bblur, mask_bblur, 0, 1, norm_type=cv2.NORM_MINMAX)
    cv2.imshow("syn", synthetic_img)
    cv2.imwrite(os.path.join(target_dir, img['file_name']), synthetic_img * 255)
    # image = cv2.imread(os.path.join(img_dir, img['file_name']))
    # cv2.imshow("img", image)
    cv2.imshow("mask", mask_bblur)
    cv2.waitKey(0)


for image_id in image_ids:
    img = coco.imgs[image_id]
    # loading annotations into memory...
    # Done (t=12.70s)
    # creating index...
    # index created!
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_ids, iscrowd=None)
    # print(np.floor(255 / len(anns_ids)))
    anns = coco.loadAnns(anns_ids)
    # binmask = coco.annToMask(anns[0]) * anns_ids[0] * np.floor(255 / len(anns_ids))
    print(os.path.join(img_dir, img['file_name']))
    originImage = cv2.imread(os.path.join(img_dir, img['file_name']), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("image", originImage)
    synthetic_img = np.zeros_like(originImage, dtype=np.float32)
    synthetic_img[:] = random.randint(-10, 10)*15
    for i in range(len(anns)):
        # binmask += coco.annToMask(anns[i]) * anns_ids[i] * np.floor(255 / len(anns_ids))
        mask = coco.annToMask(anns[i])
        intensity_variation = random.randint(-10, 10)*15
        synthetic_img[mask > 0] = intensity_variation

    # cv2.imshow("mask before blur", synthetic_img.astype(np.uint8))
    synthetic_img = cv2.GaussianBlur(synthetic_img, (11, 11), 0)
    mask_bblur = synthetic_img
    # cv2.imshow("mask after blur", synthetic_img)
    # cv2.waitKey(0)
    print(originImage.dtype)
    synthetic_img = originImage.astype(np.float32) + synthetic_img
    cv2.normalize(synthetic_img, synthetic_img, 0, 1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(mask_bblur, mask_bblur, 0, 1, norm_type=cv2.NORM_MINMAX)
    # cv2.imshow("syn", synthetic_img)
    cv2.imwrite(os.path.join(target_dir, img['file_name']), synthetic_img * 255)
    # image = cv2.imread(os.path.join(img_dir, img['file_name']))
    # cv2.imshow("img", image)
    # cv2.imshow("mask", mask_bblur)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
