import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def find_match(img1, img2):
    sift_img1 = cv2.SIFT_create()
    kp_img1, des_img1 = sift_img1.detectAndCompute(img1,None)

    sift_img2 = cv2.SIFT_create()
    kp_img2, des_img2 = sift_img2.detectAndCompute(img2,None)

    dis, ind = NearestNeighbors(n_neighbors=2).fit(des_img2).kneighbors(des_img1)

    x1, x2 = np.empty((1,2)), np.empty((1,2))

    for i in range(len(dis)):
        if dis[i][0]/dis[i][1] <= 0.7:
            x1 = np.concatenate((x1, np.array([kp_img1[i].pt[0], kp_img1[i].pt[1]]).reshape((1,2))),axis=0)
            x2 = np.concatenate((x2, np.array([kp_img2[ind[i][0]].pt[0], kp_img2[ind[i][0]].pt[1]]).reshape((1,2))),axis=0)

    x1 = x1[1:]
    x2 = x2[1:]
    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    l = len(x1)
    inl_m = 0
    for i in range(ransac_iter):
        r_ind = np.random.choice(l, 3, replace=False)
        x1_s = x1[r_ind]
        x2_s = x2[r_ind]

        A1 = np.zeros((3,3))
        for i in range(3):
            A1[i][0] = x1_s[i][0]
            A1[i][1] = x1_s[i][1]
            A1[i][2] = 1

        try:
            h = np.matmul(np.linalg.inv(A1),x2_s)
        except np.linalg.LinAlgError:
            continue
        h = np.append(h.T, [[0, 0, 1]], axis=0)
        x1_tmp = np.append(x1, np.ones((len(x1),1)), axis=1)
        x2_tmp = np.append(x2, np.ones((len(x2),1)), axis=1)
        tmp = np.matmul(x1_tmp, h.T)
        distance = np.linalg.norm(tmp - x2_tmp, axis=1)
        inl = len([x for x in distance if x<ransac_thr])
        if inl > inl_m:
            inl_m = inl
            A = h
    return A


def warp_image(img, A, output_size):
    h = output_size[0]
    w = output_size[1]
    img_warped = np.zeros(output_size)

    for i in range(h):
        for j in range(w):
            tmp = np.matmul(A, np.array([j, i, 1]))
            tmp_s = np.floor(tmp)
            img_warped[i][j] = img[np.int(tmp_s[1]), np.int(tmp_s[0])]
    return img_warped


def visualize_sift(img):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sqrt(np.sum(np.square(x2_t[:, :2] - x2), axis=1))
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y', linewidth=3)
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_warp_image(img_warped, img):
    plt.subplot(131)
    plt.imshow(img_warped, cmap='gray')
    plt.title('Warped image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(np.abs(img_warped - img), cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./zhixuan_template.jpg', 0)  # read as grey scale image
    target = cv2.imread('./zhixuan_target.jpg', 0)  # read as grey scale image

    x1, x2 = find_match(template, target)
    visualize_find_match(template, target, x1, x2)

    ransac_thr = 0.01  # specify error threshold for RANSAC (unit: pixel)
    ransac_iter = 1100  # specify number of iterations for RANSAC
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target, x1, x2, A, ransac_thr)

    img_warped = warp_image(target, A, template.shape)
    visualize_warp_image(img_warped, template)





