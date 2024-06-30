from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    # Your implementation goes here!
    n = len(dataset)
    S = np.dot(np.transpose(dataset), dataset)
    return S / (n - 1)


def get_eig(S, m):
    # Your implementation goes here!
    d = len(S)
    Lambda, U = eigh(S, subset_by_index=[d-m, d-1])
    return np.diag(Lambda[::-1]), U[:, ::-1]


def get_eig_prop(S, prop):
    # Your implementation goes here!
    Lambda, U = eigh(S)
    Lambda = Lambda[::-1]
    U = U[:, ::-1]
    lambda_prop = Lambda / np.sum(Lambda)
    num_to_keep = np.sum(lambda_prop > prop)
    return np.diag(Lambda[:num_to_keep]), U[:, :num_to_keep]


def project_image(image, U):
    # Your implementation goes here!
    a = np.dot(np.transpose(U), image)
    return np.dot(U, a)


def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 3), ncols=2)
    ax1.set_title("Original")
    im1 = ax1.imshow(orig.reshape(32, 32).T, aspect='equal', cmap='viridis')
    fig.colorbar(im1, ax=ax1)

    ax2.set_title("Projection")
    im2 = ax2.imshow(proj.reshape(32, 32).T, aspect='equal', cmap='viridis')
    fig.colorbar(im2, ax=ax2)
    return fig, ax1, ax2


if __name__ == "__main__":
    x = load_and_center_dataset("YaleB_32x32.npy")
    S = get_covariance(x)
    Lambda, U = get_eig(S, 2)
    projection = project_image(x[0], U)
    fig, ax1, ax2 = display_image(x[0], projection)
    plt.show()