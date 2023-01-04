import argparse
import copy
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


def read_kps(folder, detector, img_num):
    s1 = folder

    keypoints = []

    kp = np.loadtxt(s1 + detector + "_kp_" + str(img_num) + ".txt", delimiter=' ')
    # np.savetxt(s1 + detector + "_kp_" + str(img_num) + ".txt", np.array(keypoints))
    for keypoint in kp:
        keypoints.append(cv2.KeyPoint(keypoint[0], keypoint[1], keypoint[2], keypoint[3], keypoint[4], int(keypoint[5]), int(keypoint[6])))
        # keypoints.append([keypoint.pt[0], keypoint.pt[1], keypoint.response, keypoint.angle, keypoint.octave])

    print(detector, img_num, np.shape(keypoints))

    return keypoints


def read_desc(folder, detector, img_num):
    s1 = folder

    descs = []
    desc = np.array(np.loadtxt(s1 + detector + "_desc_" + str(img_num) + ".txt", delimiter=' '), dtype=np.float32)

    descs.append(desc)

    return descs


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
                                                 keep_k_points=1200):
    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

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


def cal_kp_dist(kp1, kp2):
    dist = np.sqrt(pow(kp1[0] - kp2[0], 2) + pow(kp1[1] - kp2[1], 2))
    return dist


def find_match_inliers(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    inliers = []
    for i in range(np.shape(matched_pts1)[0]):
        dist = cal_kp_dist(matched_pts1[i], matched_pts2[i])
        if dist < 5:
            inliers.append(1)
        else:
            inliers.append(0)

    return np.array(inliers)


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    k = np.array([[491.10682706595435, 0, 336.7371880176229],
                  [0, 490.82849146678325, 251.80134395960215],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.array([-0.40288158398373014, 0.15663066706750828, -0.000923172204407788, -0.0037779831664445556, 0],
                    dtype=np.float32)
    img = cv2.undistort(img, k, dist)

    img = cv2.resize(img, img_size)
    img = img[128:, :]
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

    kp_folder = "/home/thamilchelvan.a/NUFRL/results/kp_log_fin/"

    for filename in sorted(os.listdir(img_folder), reverse=False):
        img, img_orig = preprocess_image(os.path.join(img_folder, filename), img_size)
        images.append(img)
        images_orig.append(img_orig)

    no_images = len(images)

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)
    weights_dir_sp = Path(weights_root_dir, "sp_v6")

    graph = tf.Graph()
    graph_sp = tf.Graph()
    sess = tf.Session(graph=graph)
    sess_sp = tf.Session(graph=graph_sp)
    tf.saved_model.loader.load(sess,
                               [tf.saved_model.tag_constants.SERVING],
                               str(weights_dir))

    tf.saved_model.loader.load(sess_sp,
                               [tf.saved_model.tag_constants.SERVING],
                               str(weights_dir_sp))

    input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
    output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

    input_img_tensor_sp = graph_sp.get_tensor_by_name('superpoint/image:0')
    output_prob_nms_tensor_sp = graph_sp.get_tensor_by_name('superpoint/prob_nms:0')
    output_desc_tensors_sp = graph_sp.get_tensor_by_name('superpoint/descriptors:0')

    no_matches_SIFT = np.zeros(no_images)
    no_matches_SIFTmod = np.zeros(no_images)
    no_matches_SP = np.zeros(no_images)
    no_matches_IP = np.zeros(no_images)
    temporal_gap = []
    total = np.zeros(no_images)

    for i in range(no_images):
        temporal_gap.append(i*0.5)
        # if 20 + i * 0.5 > 23.5:
        #     temporal_gap.append(datetime.datetime(2021, 10, 21, int(20 + np.floor(i / 2) - 24), int(30 * (i % 2))))
        # else:
        #     temporal_gap.append(datetime.datetime(2021, 10, 20, int(20 + np.floor(i / 2)), int(30 * (i % 2))))
    temporal_gap = np.array(temporal_gap)

    kp_SP = []
    desc_SP = []
    kp_IP = []
    desc_IP = []
    kp_SIFT = []
    desc_SIFT = []
    kp_SIFTmod = []
    desc_SIFTmod = []

    matches_SP = []
    matches_IP = []
    matches_SIFT = []
    matches_SIFTmod = []

    for i in range(no_images):
        out = sess.run([output_prob_nms_tensor, output_desc_tensors],
                       feed_dict={input_img_tensor: np.expand_dims(images[i], 0)})
        keypoint_map = np.squeeze(out[0])
        descriptor_map = np.squeeze(out[1])
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
            keypoint_map, descriptor_map, keep_k_best)
        kp_IP.append(kp1)
        desc_IP.append(desc1)

        out_sp = sess_sp.run([output_prob_nms_tensor_sp, output_desc_tensors_sp],
                             feed_dict={input_img_tensor_sp: np.expand_dims(images[i], 0)})
        keypoint_map = np.squeeze(out_sp[0])
        descriptor_map = np.squeeze(out_sp[1])
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
            keypoint_map, descriptor_map, keep_k_best)
        kp_SP.append(kp1)
        desc_SP.append(desc1)

        # sift_kp1, sift_desc1 = extract_ORB_keypoints_and_descriptors(images_orig[i])
        # kp_SIFT.append(sift_kp1)
        # desc_SIFT.append(sift_desc1)
        sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(images_orig[i])
        kp_SIFT.append(sift_kp1)
        desc_SIFT.append(sift_desc1)

        # sift_kp1, sift_desc1, sift_mod_kp1, sift_mod_desc1 = extract_SIFT_mod_keypoints_and_descriptors(images_orig[i])
        # kp_SIFTmod.append(sift_mod_kp1)
        # desc_SIFTmod.append(sift_mod_desc1)
        kp_SIFT_mod = read_kps(kp_folder, "SIFT_mod", i)
        kp_SIFTmod.append(kp_SIFT_mod)

        desc_mod = np.array(np.squeeze(read_desc(kp_folder, "SIFT_mod", i)), dtype=np.float32)
        desc_SIFTmod.append(desc_mod)

        # kp_mod = []
        # desc_mod = []
        #
        # for kp in sift_kp1:
        #     kp_mod.append(modify_kp(kp))
        # for d in sift_desc1:
        #     desc_mod.append(np.array(modify_desc(d)))
        # kp_SIFTmod.append(kp_mod)
        # desc_SIFTmod.append(desc_mod)
        # print(np.shape(desc_mod), np.shape(desc_SIFTmod))
        # print(np.shape(sift_desc1), np.shape(desc_SIFT))

    for i in [0]:
        for j in range(1, no_images):
            # Match and get rid of outliers
            m_kp1, m_kp2, match = match_descriptors(kp_SP[i], desc_SP[i], kp_SP[j], desc_SP[j])
            # H, inliers = compute_homography(m_kp1, m_kp2)
            # F, inliers = compute_fundamental(m_kp1, m_kp2)
            inliers = find_match_inliers(m_kp1, m_kp2)

            # Draw SuperPoint matches
            t_m = len(match)
            match = np.array(match)[inliers.astype(bool)].tolist()
            matches_SP.append(match)
            no_matches_SP[abs(j - i)] += len(match)
            print("SuperPoint_matches: ", len(match), len(match)/t_m)
            total[abs(j - i)] += 1
            matched_img = cv2.drawMatches(images_orig[i], kp_SP[i], images_orig[j], kp_SP[j], match,
                                          None, matchColor=(0, 255, 0),
                                          singlePointColor=(0, 0, 255))

            cv2.imshow("SuperPoint matches", matched_img)
            cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/SuperPoint_static_latest/match00" + str(j) + ".jpg",
                        matched_img)

            m_kp1, m_kp2, match = match_descriptors(kp_IP[i], desc_IP[i], kp_IP[j], desc_IP[j])
            # H, inliers = compute_homography(m_kp1, m_kp2)
            # F, inliers = compute_fundamental(m_kp1, m_kp2)
            inliers = find_match_inliers(m_kp1, m_kp2)

            # Draw SuperPoint matches
            t_m = len(match)
            match = np.array(match)[inliers.astype(bool)].tolist()
            matches_IP.append(match)
            no_matches_IP[abs(j - i)] += len(match)
            print("IRPoint_matches: ", len(match), len(match)/t_m)
            matched_img = cv2.drawMatches(images_orig[i], kp_IP[i], images_orig[j], kp_IP[j], match,
                                          None, matchColor=(0, 255, 0),
                                          singlePointColor=(0, 0, 255))

            cv2.imshow("IRPoint matches", matched_img)
            cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/IRPoint_static_latest/match00" + str(j) + ".jpg",
                        matched_img)
            # Compare SIFT matches

            sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
                kp_SIFT[i], desc_SIFT[i], kp_SIFT[j], desc_SIFT[j])
            # sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
            # sift_F, sift_inliers = compute_fundamental(sift_m_kp1, sift_m_kp2)
            sift_inliers = find_match_inliers(sift_m_kp1, sift_m_kp2)

            # Draw SIFT matches
            t_m = len(sift_matches)
            sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
            matches_SIFT.append(sift_matches)
            no_matches_SIFT[abs(i - j)] += len(sift_matches)
            print("SIFT_matches: ", len(sift_matches), len(sift_matches)/t_m)
            sift_matched_img = cv2.drawMatches(images_orig[i], kp_SIFT[i], images_orig[j],
                                               kp_SIFT[j], sift_matches, None,
                                               matchColor=(0, 255, 0),
                                               singlePointColor=(0, 0, 255))
            cv2.imshow("SIFT matches", sift_matched_img)
            cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/SIFT_static_latest/match00" + str(j) + ".jpg",
                        sift_matched_img)

            siftmod_m_kp1, siftmod_m_kp2, siftmod_matches = match_descriptors(
                kp_SIFTmod[i], np.squeeze(desc_SIFTmod[i]), kp_SIFTmod[j], np.squeeze(desc_SIFTmod[j]))
            # siftmod_H, siftmod_inliers = compute_homography(siftmod_m_kp1, siftmod_m_kp2)
            # siftmod_F, siftmod_inliers = compute_fundamental(siftmod_m_kp1, siftmod_m_kp2)
            siftmod_inliers = find_match_inliers(siftmod_m_kp1, siftmod_m_kp2)

            # Draw SIFT matches
            t_m = len(siftmod_matches)
            siftmod_matches = np.array(siftmod_matches)[siftmod_inliers.astype(bool)].tolist()
            matches_SIFTmod.append(siftmod_matches)
            no_matches_SIFTmod[abs(i - j)] += len(siftmod_matches)
            print("modSIFT_matches: ", len(siftmod_matches), len(siftmod_matches)/t_m)
            siftmod_matched_img = cv2.drawMatches(images_orig[i], kp_SIFTmod[i], images_orig[j],
                                               kp_SIFTmod[j], siftmod_matches, None,
                                               matchColor=(0, 255, 0),
                                               singlePointColor=(0, 0, 255))
            cv2.imshow("SIFTmod matches", siftmod_matched_img)
            cv2.imwrite("/home/thamilchelvan.a/NUFRL/results/SIFTmod_static_latest/match00" + str(j) + ".jpg",
                        siftmod_matched_img)

            cv2.waitKey(0)

        fig, ax = plt.subplots(1)
        fig.autofmt_xdate()
        plt.plot(temporal_gap, no_matches_SIFT / total, 'r-', temporal_gap, no_matches_SIFTmod / total, 'c-')
        # xfmt = mdates.DateFormatter('%H:%M')
        # ax.xaxis.set_major_formatter(xfmt)
        plt.legend(['SIFT', 'modified SIFT'])
        ax.yaxis.set_major_locator(MultipleLocator(100))
        plt.minorticks_on()
        plt.grid(which='both')
        plt.title('Number of Matches V Temporal Gap')
        plt.xlabel('Temporal gap (hours)')
        plt.ylabel('number of inlier matches')
        plt.show()
