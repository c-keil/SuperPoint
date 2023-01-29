import argparse
from pathlib import Path

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import glob
import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import datetime
from copy import deepcopy
from shutil import copy2

from superpoint.settings import EXPER_PATH  # noqa: E402


def extract_SIFT_keypoints_and_descriptors(img, n_features=1200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(n_features)
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def modify_SIFT_kp(keypoints):
    for keypoint in keypoints:
        keypoint.angle = 0
        # if keypoint.angle >= 180:
        #     keypoint.angle -= 180
    return keypoints


def modify_SIFT_desc(desc):
    desc_new = desc.copy()
    for i in range(16):
        # bin = desc[i*8:(i+1)*8]
        for j in range(i*8, (i+1)*8):
            if j%8 < 4:
                desc_new[j] = (desc[j] + desc[j+4]) / 2
            else:
                desc_new[j] = (desc[j] + desc[j - 4]) / 2
    return desc_new


def extract_SIFT_mod_keypoints_and_descriptors(img, n_features=1200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(n_features)
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    kp_mod = modify_SIFT_kp(kp)
    kp_mod, desc_mod = sift.compute(np.squeeze(gray_img), kp_mod)

    for i in range(np.shape(desc)[0]):
        desc_mod[i, :] = modify_SIFT_desc(desc_mod[i, :])
    return kp, desc, kp_mod, desc_mod


def extract_SIFT_mod_descriptors(img, kp, kp_mod, n_features=1200):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(n_features)
    # kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    # kp_mod = modify_SIFT_kp(kp)
    kp, desc = sift.compute(np.squeeze(gray_img), kp)
    kp_mod, desc_mod = sift.compute(np.squeeze(gray_img), kp_mod)

    for i in range(np.shape(desc_mod)[0]):
        desc_mod[i, :] = modify_SIFT_desc(desc_mod[i, :])
    return kp, desc, kp_mod, desc_mod


def extract_harris_keypoints_and_descriptors(img):
    kp = []
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    harris_ = harris>0.01*harris.max()
    for i in range(np.shape(harris_)[0]):
        for j in range(np.shape(harris_)[1]):
            if harris_[i, j]:
                key_point = cv2.KeyPoint()
                key_point.pt = (j, i)
                key_point.response = harris[i, j]
                kp.append(key_point)
    return kp


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

def select_k_best(points, k):
    """ Select the k most probable points (and strip their proba).
    points has shape (num_points, 3) where the last coordinate is the proba. """
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
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


def compute_fundamental(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    F, inliers = cv2.findFundamentalMat(matched_pts1[:, [1, 0]],
                                        matched_pts2[:, [1, 0]],
                                        cv2.FM_RANSAC, 3, 0.99)
    inliers = inliers.flatten()
    return F, inliers


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    # img = img[128:, :]
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig


def log_data(kp, desc, detector, image_file_path):

    #get file name
    image_name = os.path.splitext(os.path.basename(image_file_path))[0]
    kp_save_path = os.path.dirname(image_file_path) + "/keypoints/"
    desc_save_path = os.path.dirname(image_file_path) + "/descriptors/"
    image_copy_path = os.path.dirname(image_file_path) + "/images/" 
    if not os.path.exists(kp_save_path):
        os.mkdir(kp_save_path)
    if not os.path.exists(desc_save_path):
        os.mkdir(desc_save_path)
    if not os.path.exists(image_copy_path):
        os.mkdir(image_copy_path)
    # f = open(s1+detector+"_kp_"+str(img_num)+".txt", 'w')
    # g = open(s1 + detector + "_desc_"+str(img_num)+".txt", 'w')
    keypoints = []
    for keypoint in kp:
        keypoints.append([keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id])
    # print(keypoints)
    # print(detector, img_num, np.shape(keypoints))
    np.savetxt(kp_save_path + image_name + "_kp.txt", np.array(keypoints))
    np.savetxt(desc_save_path + image_name + "_desc.txt", desc)
    copy2(image_file_path, os.path.join(image_copy_path, image_name + ".png"))
    
    # input("enter to continue?")
    # np.savetxt(s1+detector+"_kp_"+str(img_num)+".txt", np.array(keypoints))
    # if desc is not None:
    #     np.savetxt(s1 + detector + "_desc_"+str(img_num)+".txt", desc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Extract Harris, SIFT and SuperPoint keypoints and descriptors to log file')
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
    im_paths = []

    for filename in sorted(glob.glob(os.path.join(img_folder,"*.png")), reverse=False):
        im_path = os.path.join(img_folder, filename)
        # print(im_path)
        if not os.path.isdir(im_path):
            img, img_orig = preprocess_image(im_path, img_size)
            images.append(img)
            images_orig.append(img_orig)
            im_paths.append(im_path)

    no_images = len(images)
    # no_images = 200

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        no_matches_SIFT = np.zeros(no_images)
        no_matches_SP = np.zeros(no_images)
        temporal_gap = []
        total = np.zeros(no_images)

        # for i in range(no_images):
        #     if(20+i*0.5 > 23.5):
        #         temporal_gap.append(datetime.datetime(2021, 10, 21, int(20 + np.floor(i / 2) - 24), int(30 * (i % 2))))
        #     else:
        #         temporal_gap.append(datetime.datetime(2021, 10, 20, int(20 + np.floor(i/2)), int(30*(i%2))))
        # temporal_gap = np.array(temporal_gap)

        kp_SP = []
        desc_SP = []
        kp_SIFT = []
        desc_SIFT = []
        kp_SIFT_mod = []
        desc_SIFT_mod = []
        kp_harris = []
        desc_harris = []

        matches_SP = []
        matches_SIFT = []

        keypoint_map = []
        prev_keypoint_map = []

        for i in range(260,no_images):
        # for i in range(no_images):
            print("processing {} of {}".format(i, no_images))
            out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                            feed_dict={input_img_tensor: np.expand_dims(images[i], 0)})
            # if i==0:
            keypoint_map = np.squeeze(out[0])
            descriptor_map = np.squeeze(out[1])
            
            # print(keypoint_map)
            # print(prev_keypoint_map == keypoint_map)
            # print(type(keypoint_map))
            prev_keypoint_map = keypoint_map
            kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
                    keypoint_map, descriptor_map, keep_k_best)
            # print("kp1: {}".format(kp1[0].pt))
            kp_SP.append(kp1)
            desc_SP.append(desc1)
            # sift_kp1, sift_desc1 = extract_ORB_keypoints_and_descriptors(images_orig[i])
            # kp_SIFT.append(sift_kp1)
            # desc_SIFT.append(sift_desc1)

            # sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(images_orig[i])
            # kp_SIFT.append(sift_kp1)
            # desc_SIFT.append(sift_desc1)

            # if i == 0:
            #     sift_kp1, sift_desc1, sift_mod_kp1, sift_mod_desc1 = extract_SIFT_mod_keypoints_and_descriptors(images_orig[i])
            # else:
            #     sift_kp1, sift_desc1, sift_mod_kp1, sift_mod_desc1 = extract_SIFT_mod_descriptors(images_orig[i], kp_SIFT[0], kp_SIFT_mod[0])

            # kp_SIFT.append(sift_kp1)
            # desc_SIFT.append(sift_desc1)

            # kp_SIFT_mod.append(sift_mod_kp1)
            # desc_SIFT_mod.append(sift_mod_desc1)

            # harris_kp1 = extract_harris_keypoints_and_descriptors(images_orig[i])
            # kp_harris.append(harris_kp1)
            # view kp
            # im1 = cv2.imread(im_paths[i])
            # print(type(images[i]))
            # print(images[i].ndim)
            # print(images[i].dtype)
            # print(images[i].shape)
            # im2 = deepcopy(im1)
            # print(type(im2))
            # print(im1.ndim)
            # print(im1.dtype)
            # print(im1.shape)
            # cv2.drawKeypoints(im1,kp1,im2)
            # cv2.imshow("kps", im2)
            # cv2.waitKey()

            log_data(np.array(kp1), desc1, "SuperPoint", im_paths[i])
            #
            # 
            #  log_data(np.array(sift_kp1), sift_desc1, "SIFT", i)
            # log_data(np.array(sift_mod_kp1), sift_mod_desc1, "SIFT_mod", i)
            # log_data(np.array(harris_kp1), None, "harris", i)

        # for i in [0]:
        #     for j in range(no_images):
        #
        #         # Match and get rid of outliers
        #         m_kp1, m_kp2, match = match_descriptors(kp_SP[i], desc_SP[i], kp_SP[j], desc_SP[j])
        #         H, inliers = compute_homography(m_kp1, m_kp2)
        #
        #         # Draw SuperPoint matches
        #         match = np.array(match)[inliers.astype(bool)].tolist()
        #         # matches_SP.append(match)
        #         no_matches_SP[abs(j-i)] += len(match)
        #         total[abs(j-i)] += 1
        #         matched_img = cv2.drawMatches(images_orig[i], kp_SP[i], images_orig[j], kp_SP[j], match,
        #                                       None, matchColor=(0, 255, 0),
        #                                       singlePointColor=(0, 0, 255))
        #
        #         cv2.imshow("SuperPoint matches", matched_img)
        #         cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/SuperPoint_IR_matches/match00"+str(j)+".jpg", matched_img)
        #
        #         # Compare SIFT matches
        #
        #         sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
        #                 kp_SIFT[i], desc_SIFT[i], kp_SIFT[j], desc_SIFT[j])
        #         sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
        #
        #         # Draw SIFT matches
        #         sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
        #         # matches_SIFT.append(sift_matches)
        #         no_matches_SIFT[abs(i-j)] += len(sift_matches)
        #         sift_matched_img = cv2.drawMatches(images_orig[i], kp_SIFT[i], images_orig[j],
        #                                            kp_SIFT[j], sift_matches, None,
        #                                            matchColor=(0, 255, 0),
        #                                            singlePointColor=(0, 0, 255))
        #         cv2.imshow("SIFT matches", sift_matched_img)
        #         cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/SIFT_IR_matches/match00" + str(j) + ".jpg",
        #                     sift_matched_img)
        #
        #         cv2.waitKey(5)

        # fig, ax = plt.subplots(1)
        # fig.autofmt_xdate()
        # plt.plot(temporal_gap, no_matches_SIFT/total, 'r-', temporal_gap, no_matches_SP/total, 'b-')
        # xfmt = mdates.DateFormatter('%H:%M')
        # ax.xaxis.set_major_formatter(xfmt)
        # plt.legend(['SIFT', 'SuperPoint'])
        # ax.yaxis.set_major_locator(MultipleLocator(400))
        # plt.minorticks_on()
        # plt.grid(which='both')
        # plt.title('Comparison of Inlier Matches Over Time SIFT Vs SuperPoint')
        # plt.xlabel('Time of day')
        # plt.ylabel('number of inlier matches')
        # plt.show()