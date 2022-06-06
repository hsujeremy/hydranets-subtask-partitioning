#!/usr/bin/env python3
from collections import deque
import math
import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions


def extract_raw_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    transformed_img = image.img_to_array(img)
    transformed_img = np.expand_dims(transformed_img, axis=0)
    transformed_img = preprocess_input(transformed_img)
    raw_features = model.predict(transformed_img)
    raw_features = np.nan_to_num(raw_features)
    return raw_features


def predict(img_path, model):
    if not os.path.isfile(img_path):
        print('Image not found!')
        return None

    features = extract_raw_features(img_path, model)
    prediction_label = decode_predictions(features, top=1)
    return prediction_label[0][0][1]


def extract_features(img_path, model):
    if not os.path.isfile(img_path):
        print('Image not found!')
        return None

    features = extract_raw_features(img_path, model)
    return features.squeeze()


def extract_average_features(dir, model, sample_size, root):
    if not os.path.isdir(dir):
        print('{} not found!'.format(dir))
        return None

    all_features = []

    images = os.listdir(dir)
    images = [img for img in images if img.endswith('.JPEG')]

    for count, img in enumerate(images):
        if count == sample_size:
            break

        img_path = os.path.join(root, 'train', dir, img)
        features = extract_features(img_path, model)
        features = np.nan_to_num(features)
        assert features.shape == (1000,)
        all_features.append(features)

    all_features = np.asarray(all_features)

    # Expect this shape for all classes since this is the output layer of
    # ImageNet
    assert all_features.shape == (sample_size, 1000)

    features_sum = all_features.sum(axis=0)
    return features_sum / sample_size


def partition(features, group_max_size):
    num_clusters = math.ceil(len(features) / group_max_size)

    kmeans_model = KMeans(n_clusters=num_clusters).fit(features)
    centroids = kmeans_model.cluster_centers_
    # print(centroids)

    items = list(enumerate(features))
    # print(items)

    groups = []

    for centroid in centroids:
        dists = []

        for key, value in items:
            dist = np.linalg.norm(centroid - value)
            dists.append((key, dist))

        dists.sort(key=lambda x: x[1])
        dists = deque(dists)
        group = []

        while dists and len(group) < group_max_size:
            current = dists.popleft()
            group.append(current)

        groups.append([key for key, _ in group])
        to_remove = {i for i, _ in group}
        items = [i for i in items if i[0] not in to_remove]

    return groups


def main():
    root = os.getcwd()
    os.chdir(root)
    model = keras.applications.ResNet50()

    print(os.getcwd())
    if not os.path.exists('train'):
        print('Training directory not found')
        sys.exit(0)

    os.chdir('train')

    dir_list = os.listdir()
    dir_list = [d for d in dir_list if os.path.isdir(d)]
    num_classes = len(dir_list)

    print('Extracting features now')

    all_features = []
    for item in dir_list:
        path = os.path.join(root, 'train', item)
        class_features = extract_average_features(path, model, 125, root)
        class_features = np.nan_to_num(class_features)
        all_features.append(class_features)
        break

    print('Finished feature extraction, moving onto partitioning')

    for max_group_size in range(2, num_classes):
        print('Partitioning classes into groups of size ' + str(max_group_size))
        all_features_cpy = [features[:] for features in all_features]
        groups = partition(all_features_cpy, max_group_size)
        print('Finished partitioning, writing out groupings')

        outpath = os.path.join(root, 'groupings',
                               'groups_{}.txt'.format(str(max_group_size)))
        with open(outpath, 'w') as outfile:
            for group in groups:
                for i in group:
                    outfile.write(dir_list[i] + ' ')
                outfile.write('\n')


if __name__ == '__main__':
    main()
