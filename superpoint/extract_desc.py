import argparse
from pathlib import Path

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import datetime

from superpoint.settings import EXPER_PATH  # noqa: E402


def extract_SIFT_keypoints_and_descriptors(img, n_features=1200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(n_features)
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def extract_SURF_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    kp, desc = surf.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def extract_ORB_keypoints_and_descriptors(img, n_features=1200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, desc = orb.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0.4)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def compute_fundamental(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    F, inliers = cv2.findFundamentalMat(matched_pts1[:, [1, 0]],
                                        matched_pts2[:, [1, 0]],
                                        cv2.FM_RANSAC, 3, 0.99)
    inliers = inliers.flatten()
    return F, inliers


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = cv2.resize(img, img_size)
    # img = img[128:, :]
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description='Compute matches between IR images with varying temporal gaps 0.5-18hr with SIFT and SuperPoint')
    parser.add_argument('weights_name', type=str)
    parser.add_argument('img_folder_path', type=str)
    parser.add_argument('--H', type=int, default=512,
                        help='The height in pixels to resize the images to. \
                                (default: 512)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1200,
                        help='Maximum number of keypoints to keep \
                        (default: 1200)')
    args = parser.parse_args()

    weights_name = args.weights_name
    img_folder = args.img_folder_path
    img_size = (args.W, args.H)
    keep_k_best = args.k_best

    images_orig = []
    images = []
    txt_file = []
    txt_file_kp = []
    for filename in sorted(os.listdir(img_folder), reverse=False):
        img, img_orig = preprocess_image(os.path.join(img_folder, filename), img_size)
        if img is not None:
            txt_file.append(os.path.join(img_folder+"txt/", filename[:-4]+".txt"))
            txt_file_kp.append(os.path.join(img_folder + "txt_kp/", filename[:-4] + ".txt"))
            images.append(img)
            images_orig.append(img_orig)

    no_images = min(len(images), 1000)

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    tf.saved_model.loader.load(sess,
                               [tf.saved_model.tag_constants.SERVING],
                               str(weights_dir))

    input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
    output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

    kp_IP = []
    desc_IP = []

    for i in range(no_images):
        out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                       feed_dict={input_img_tensor: np.expand_dims(images[i], 0)})
        keypoint_map = np.squeeze(out[0])
        descriptor_map = np.squeeze(out[1])
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
            keypoint_map, descriptor_map, keep_k_best)
        kp_IP.append(kp1)
        desc_IP.append(desc1)
        kp = np.array([[k.pt[0], k.pt[1]] for k in kp1])
        print(np.shape(kp))
        np.savetxt(txt_file[i], np.array(desc1))
        np.savetxt(txt_file_kp[i], kp)


