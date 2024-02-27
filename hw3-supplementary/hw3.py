#Name: David Li
#Email: sli857@wisc.edu
#NetID: sli857
#CS Login: sli857

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
N = 2414
D = 1024
def load_and_center_dataset(filename):
    dataset = np.load(filename)
    #print(dataset.shape)
    means = np.mean(dataset, axis = 0)
    dataset = dataset - means
    return dataset

def get_covariance(dataset):
    x = np.dot(np.transpose(dataset), dataset)
    return x / (N - 1)

def get_eig(S, m):
    # Your implementation goes here!
    eigvals, eigvecs = eigh(S, subset_by_index=[len(S) - m, len(S) - 1], eigvals_only=False)
    eigvecs = np.fliplr(eigvecs)
    eigvals = -np.sort(-eigvals)
    eigvals = np.diag(eigvals)
    return eigvals, eigvecs

def get_eig_prop(S, prop):
    # Your implementation goes here!
    eigvals = eigh(S, eigvals_only=True)
    min = prop * sum(eigvals)
    eigvals, eigvecs = eigh(S, subset_by_value = [min, np.inf])
    eigvecs = np.fliplr(eigvecs)
    eigvals = -np.sort(-eigvals)
    eigvals = np.diag(eigvals)
    return eigvals, eigvecs

def project_image(image, U):
    # Your implementation goes here!
    projection = np.dot(np.transpose(U), image)
    reconstruciton = np.dot(U, projection)
    return reconstruciton

def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    origReshaped = np.reshape(orig, (32, 32))
    origReshaped = np.rot90(np.fliplr(origReshaped))
    projReshaped = np.reshape(proj, (32, 32))
    projReshaped = np.rot90(np.fliplr(projReshaped))
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    ax1.imshow(origReshaped, aspect = "equal")
    ax1.set_title("Original")
    ax2.imshow(projReshaped, aspect = "equal")
    ax2.set_title("Projection")
    fig.colorbar(ax1.images[0], ax = ax1)
    fig.colorbar(ax2.images[0], ax = ax2)
    fig.canvas.draw()
    return fig, ax1, ax2

def main():
    #dataset = load_and_center_dataset('./hw3/hw3-supplementary/YaleB_32x32.npy')
    dataset = load_and_center_dataset('./YaleB_32x32.npy')
    #print(len(dataset))
    #print(len(dataset[0]))

    S = get_covariance(dataset)

    Lambda, U = get_eig(S, 2)
    print(Lambda)
    print(U)

    # Lambda, U = get_eig_prop(S, 0.07)
    # print(Lambda)
    # print(U)

    projection = project_image(dataset[0], U)
    print(projection)

    fig, ax1, ax2 = display_image(dataset[0], projection)
    plt.show()

if __name__ == '__main__':
    main()
