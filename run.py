import numpy as np
import os
import sys


# Directory Settings
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VECTORS_DIR = os.path.join(BASE_DIR, 'binary_vectors')


# Algorithm Parameters
NUMBER_OF_CLUSTERS = 25  # this should be a full square for diamond neighborhood
INITIAL_LEARNING_RATE = 0.6
FINAL_LEARNING_RATE = 0.01
EPOCHS = 100


def _get_input_vector(input_file_path):
    input_list = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file.readlines():
            line = line.strip()
            if line:
                input_strings = line.split()
                for input_string in input_strings:
                    input_list.append(int(input_string))

    return np.array(input_list)


def _get_input_vectors():
    input_labels = []
    input_vectors = []

    label_names = os.listdir(VECTORS_DIR)
    for label_name in label_names:
        label_dir = os.path.join(VECTORS_DIR, label_name)
        for input_file_name in os.listdir(label_dir):
            input_file_path = os.path.join(label_dir, input_file_name)
            input_vector = _get_input_vector(input_file_path)

            input_labels.append(label_name + input_file_name.replace('.txt', ''))
            input_vectors.append(input_vector)

    return input_labels, input_vectors


def _init_weight_vectors(input_vector_size):
    weight_vectors = []
    for i in range(NUMBER_OF_CLUSTERS):
        weight_vectors.append(np.random.rand(input_vector_size))
    return weight_vectors


def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def _get_2d_index(index, total_size):
    return index // isqrt(total_size),\
           index % isqrt(total_size),\
           isqrt(total_size) - 1,\
           isqrt(total_size) - 1


def _get_1d_index(x, y, total_size):
    return y * isqrt(total_size) + x


def _get_neighborhood_indices(weight_vectors, winner_index, mode):
    neighborhood_indices = [winner_index]

    if mode == 'none':
        pass
    elif mode == 'linear':
        if winner_index > 0:
            neighborhood_indices.append(winner_index - 1)
        if winner_index < len(weight_vectors) - 1:
            neighborhood_indices.append(winner_index + 1)
    elif mode == 'diamond':
        x, y, max_x, max_y = _get_2d_index(winner_index, NUMBER_OF_CLUSTERS)
        if x > 0:
            neighborhood_indices.append(_get_1d_index(x - 1, y, NUMBER_OF_CLUSTERS))
        if x < max_x:
            neighborhood_indices.append(_get_1d_index(x + 1, y, NUMBER_OF_CLUSTERS))
        if y > 0:
            neighborhood_indices.append(_get_1d_index(x, y - 1, NUMBER_OF_CLUSTERS))
        if y < max_y:
            neighborhood_indices.append(_get_1d_index(x, y + 1, NUMBER_OF_CLUSTERS))
    else:
        print('Unknown mode, defaulting to none')

    return neighborhood_indices


def _learn_weight_vectors(input_vectors, mode):
    input_vector_size = len(input_vectors[0])
    weight_vectors = _init_weight_vectors(input_vector_size)

    alpha = INITIAL_LEARNING_RATE
    for i in range(EPOCHS):
        for input_vector in input_vectors:
            min_distance = sys.float_info.max
            winner_index = -1
            for j, weight_vector in enumerate(weight_vectors):
                distance = np.linalg.norm(input_vector - weight_vector)
                if distance < min_distance:
                    min_distance = distance
                    winner_index = j
            neighborhood_indices = _get_neighborhood_indices(weight_vectors, winner_index, mode)
            for neighborhood_index in neighborhood_indices:
                weight_vectors[neighborhood_index] = alpha * input_vector + (1 - alpha) * weight_vectors[neighborhood_index]

        alpha = INITIAL_LEARNING_RATE - (i + 1) * (INITIAL_LEARNING_RATE - FINAL_LEARNING_RATE) / EPOCHS

    return weight_vectors


def _print_clusters(input_labels, input_vectors, weight_vectors):
    clusters = {}
    for i, input_vector in enumerate(input_vectors):
        min_distance = sys.float_info.max
        min_index = -1
        for j, weight_vector in enumerate(weight_vectors):
            distance = np.linalg.norm(input_vector - weight_vector)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        if min_index in clusters:
            clusters[min_index].append(input_labels[i])
        else:
            clusters[min_index] = [input_labels[i]]

    for cluster_number, cluster_content in clusters.items():
        if cluster_content:
            print(' '.join(cluster_content))


def run(mode):
    input_labels, input_vectors = _get_input_vectors()
    weight_vectors = _learn_weight_vectors(input_vectors, mode)
    _print_clusters(input_labels, input_vectors, weight_vectors)


if __name__ == '__main__':
    mode_argument = sys.argv[1]
    run(mode_argument)
