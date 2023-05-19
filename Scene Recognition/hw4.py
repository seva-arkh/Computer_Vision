import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from scipy import stats
from pathlib import Path, PureWindowsPath
import os
from tqdm import tqdm


def get_tiny_image(img, tiny_size):

    h_s = img.shape[0]//tiny_size[0] + 1
    w_s = img.shape[1]//tiny_size[1] + 1
    s = h_s * w_s

    feature = np.zeros((tiny_size[0],tiny_size[1]))

    for i in range(tiny_size[0]):
      for j in range(tiny_size[1]):
        feature[i][j] = np.sum(img[i*h_s:(i+1)*h_s][j*w_s:(j+1)*w_s])/s


    feature = np.divide(feature - np.mean(feature), np.sqrt(np.sum(feature**2)))
    feature = np.reshape(feature, (tiny_size[0]*tiny_size[1], 1))
    return feature


def predict_knn(feature_train, label_train, feature_test, n_neighbors):
    dis, ind = NearestNeighbors(n_neighbors=n_neighbors).fit(feature_train).kneighbors(feature_test, n_neighbors=n_neighbors)
    pred_test = []
    for i in range(dis.shape[0]):
      pred_test.append(label_train[ind[i][np.argmax(dis[i])]])
    pred_test = np.asarray(pred_test)
    return pred_test


def compute_confusion_matrix_and_accuracy(pred, label, n_classes):
    confusion = np.zeros((n_classes, n_classes))
    t =0 
    for i in range(pred.shape[0]):
        if pred[i]==label[i]:
            t+=1
        confusion[pred[i], label[i]]+=1   
    accuracy = t/label.shape[0]
    return confusion, accuracy


def compute_dsift(img, stride, size):
    h_i, w_i = np.shape(img)
    h_s = (h_i - size)//stride+1
    w_s = (w_i - size)//stride+1
    dsift = np.empty((0,128))
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range (h_s):
        for j in range (w_s):
            f = img[i*stride:(i*stride)+size] [j*stride:(j*stride)+size]
            k = [cv2.KeyPoint(j*stride + size//2, i*stride + size//2, size)]
            features, des = sift.compute(img, k)
            dsift = np.append(dsift, des, axis=0)
    return dsift


def build_visual_dictionary(features, dict_size):
    vocab = KMeans(n_clusters=dict_size, n_init=5,  max_iter=1500).fit(features).cluster_centers_
    return vocab


def compute_bow(dsift, vocab):
    dis, ind = NearestNeighbors().fit(vocab).kneighbors(dsift, n_neighbors=1)
    bow_feature = np.zeros((vocab.shape[0], 1))
    for i in ind:
      bow_feature[i[0]][0] += 1
    bow_feature = np.divide(bow_feature - np.mean(bow_feature), np.sqrt(np.sum(bow_feature**2)))
    return list(bow_feature)


def predict_svm(feature_train, label_train, feature_test, n_classes):
    clf = LinearSVC(C=8)
    pred = np.zeros((feature_test.shape[0], n_classes))
    models = []
    pred_test = []

    for i in range(n_classes):
        l = []
        for c in label_train:
            if c == i:
                l.append(1)
            else:
                l.append(0)
        model = clf.fit(feature_train, l)
        models.append(model)
        pred[:,i] = model.decision_function(feature_test)

    for i in range(len(feature_test)):
        pred_test.append(np.argmax(pred[i]))
    return np.array(pred_test)


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, image_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        image_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, image_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        image_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, image_train_list, label_test_list, image_test_list


def get_scene_classification_data(data_dir):
    label_classes, label_train_list, image_train_list, label_test_list, image_test_list = \
        extract_dataset_info(data_dir)

    image_train, label_train, image_test, label_test = [], [], [], []
    for i, img_path in enumerate(image_train_list):
        image_train.append(cv2.imread(img_path, 0))
        label_train.append(label_classes.index(label_train_list[i]))
    for i, img_path in enumerate(image_test_list):
        image_test.append(cv2.imread(img_path, 0))
        label_test.append(label_classes.index(label_test_list[i]))

    label_train = np.array(label_train).reshape((-1, 1))
    label_test = np.array(label_test).reshape((-1, 1))

    return image_train, label_train, image_test, label_test, label_classes


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Data preparation
    image_train, label_train, image_test, label_test, label_classes = get_scene_classification_data(
        'scene_classification_data')

    # Tiny + KNN
    feature_train = np.hstack([get_tiny_image(img, (16, 16)) for img in image_train]).T  # (1500, 256)
    feature_test = np.hstack([get_tiny_image(img, (16, 16)) for img in image_test]).T  # (1500, 256)
    n_neighbors = 5
    pred_test = predict_knn(feature_train, label_train, feature_test, n_neighbors)  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)


    # Bag-of-words + KNN / SVM

    # Bag-of-words
    print(1)
    # 1. extract dense sift features
    stride, keypoint_size = 16, 16
    dsift_train = [compute_dsift(image, stride, keypoint_size) for image in tqdm(
        image_train, 'Extracting dense SIFT features for images from train set')]  # a list of (n, 128)
    dsift_test = [compute_dsift(image, stride, keypoint_size) for image in tqdm(
        image_test, 'Extracting dense SIFT features for images from test set')]  # a list of (n, 128)

    print(2)
    # 2. build dictionary from train data
    dic_size = 50
    vocab = build_visual_dictionary(np.vstack(dsift_train), dic_size)

    print(3)
    # 3. extract bag-of-words features
    feature_train = np.hstack([compute_bow(dsift, vocab) for dsift in dsift_train]).T  # (n_train, dic_size)
    feature_test = np.hstack([compute_bow(dsift, vocab) for dsift in dsift_test]).T  # (n_test, dic_size)

    print(4)
    # KNN
    n_neighbors = 5
    pred_test = predict_knn(feature_train, label_train, feature_test, n_neighbors)  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    print(5)
    # SVM
    pred_test = predict_svm(feature_train, label_train, feature_test, len(label_classes))  # (1500, 1)
    confusion, accuracy = compute_confusion_matrix_and_accuracy(pred_test, label_test, len(label_classes))
    visualize_confusion_matrix(confusion, accuracy, label_classes)

