import click
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


def cost_function(x_true, x_em):
    """
    :param x_true: The pixels of original image (normalized). Each row is an RGB vector.
    :param x_em: The image that comes from the training of em.
    :return: The error between the real and the segmented image.
    """

    n = x_true.shape[0]  # The The number of our examples (pixels)

    return 1/n * np.sum(np.linalg.norm(x_true - x_em)**2)


def construct_image(height, width, g, m):
    """
    :param height: The height of the image
    :param width:  The width of the image
    :param g: Table (N x K). Contains the posterior probabilities of each example n belongs to each one
    of the K categories-segments.
    :param m: Table (K x 3). Contains an average vector (RGB color) from the data that belongs
    on the k-th category-segment
    :return: The normalized image which is needed to compute the error of the expectation maximisation algorithm and
    the colored image which is produced from the algorithm.
    """
    new_image = np.zeros((height * width, 3))

    for n in range(g.shape[0]):
        k = g[n].argmax()
        new_image[n] = m[k]

    flat = deepcopy(new_image)

    new_image = new_image.reshape((height, width, 3))

    k = m.shape[0]

    plt.imshow(new_image)
    plt.savefig('em_{}'.format(k))
    plt.show()

    return flat, new_image


def gaussian_mixture(x, p, m, s, k):
    """
    :param x: Our data (pixels). Each row is an RGB vector of a pixel
    :param p: Table (N x K). Contains the prior probabilities of each example n belongs to each category-segment k
    :param m: Table (K x 3). Contains an average vector(RGB) of the color from the data that belongs
    on category-segment k
    :param s: The covariance table S.
    :param k: The number of the categories.
    :return:  Table of shape (N x K) that contains the probabilities (that comes for mixture of Gaussian distributions)
    for each example n (pixel) belongs to each one of the K categories
    """
    probabilities = np.zeros((x.shape[0], k))

    for k_i in range(k):

        first_part = 1 / np.sqrt(2 * np.pi * s[k_i])
        second_part = np.exp(-(1 / (2 * s[k_i])) * (x - m[k_i, :]) ** 2)

        probabilities[:, k_i] = p[k_i] * np.prod(first_part * second_part, axis=1)

    return np.array(probabilities)


def log_likelihood(probabilities):
    return np.sum(np.log(np.sum(probabilities, axis=1)))


def maximization_step(x, g):
    """
    Execute the maximization stem of the algorithm and update the parameters
    :param x: Our data (pixels). Each row is an RGB vector of a pixel
    :param g: Table (N x K). Contains the posterior probabilities of each example n belongs to each one of
    the K categories-segments
    :return: The updated parameters.
    p: Table (N x K). Contains the prior probabilities of each example n belongs to each category-segment k
    m: Table (K x 3). Contains an average vector(RGB) of the color from the data that belongs on category-segment k
    s: The covariance table S.
    """

    k = g.shape[1]

    m = np.zeros((k, x.shape[1]))
    p = np.zeros(k)
    s = np.zeros(k)

    for k_i in range(k):
        g_k = g[:, k_i].reshape((-1, 1))

        m[k_i, :] = np.sum(g_k * x, axis=0) / np.sum(g_k)

        s[k_i] = np.sum(np.sum(g_k * ((x - m[k_i]) ** 2), axis=1)) / (x.shape[1] * np.sum(g_k))

        p[k_i] = np.sum(g_k) / x.shape[0]

    return p, m, s


def expectation_step(probabilities):
    """
    Execute the expectation step of the algorithm
    :param probabilities: Table (N x K). Contains the probabilities (that comes for Gaussian mixture)
    of each example n belongs to each category-segment k
    :return: Table (N x K). Contains the posterior probabilities of each example n belongs to each one of
    the K categories-segments
    """

    denominator = np.sum(probabilities, axis=1)

    return probabilities / denominator.reshape((-1, 1))


def initialize_parameters(k, d):
    """
    :param k: The number of categories-segments
    :param d: The dimension of each example-pixel (R, G, B) = 3
    :return: The prior probabilities, the average_vectors and the covariance table
    """

    # At the start, the prior probability of each category is the same and equal to 1/k
    prior = np.full(k, 1/k)

    # We have average_vector of d-dimension for each category and initialize
    # them with values from 0-1 because we have a normalized image.
    m = np.zeros((k, d))

    for i in range(m.shape[0]):
        m[i, :] = np.random.uniform(0, 1, d)

    # Initialize the covariance of each category
    # with values between 0.2-0.8 -> 60% of the real values
    s = np.random.uniform(0.2, 0.8, k)

    return prior, m, s


def expectation_maximization(x, k):
    """
    :param x: Table dimension (N x 3) with the pixels of th image
    :param k: The number o categories-segments
    :return: The probabilities of N example belongings on the k category and
    the average vectors of each category
    """
    d = x.shape[1]  # The dimension of each example-pixel (R G B) = 3
    tolerance = 1e-6

    prior, m, s = initialize_parameters(k, d)

    prob = gaussian_mixture(x, prior, m, s, k)

    for t in range(100):

        log_likelihood_old = log_likelihood(prob)

        g = expectation_step(prob)

        prior, m, s = maximization_step(x, g)

        prob = gaussian_mixture(x, prior, m, s, k)

        log_likelihood_new = log_likelihood(prob)
        print('log_likelihood of {:<3} iteration: {}'.format(t, log_likelihood_new))

        if log_likelihood_new - log_likelihood_old < 0:
            print('Error in coding')

        if np.abs(log_likelihood_new - log_likelihood_old) < tolerance:
            print('Converged')
            break

    return g, m


@click.command()
@click.option('--segments', default=8)
@click.option('--path', default='im.jpg')
def main(segments, path):
    img = plt.imread(path)

    print('Image shape: {}'.format(img.shape))

    #  Calculate our N independent pixels
    number_of_pixels = img.shape[0] * img.shape[1]

    #  Reshape the image in order to have a table (N x 3)
    data = img.reshape((number_of_pixels, 3))

    # Normalize our data
    data = data / 255

    post_probabilities, average_vectors = expectation_maximization(x=data, k=segments)
    flt, new_img = construct_image(img.shape[0], img.shape[1], post_probabilities, average_vectors)

    error = cost_function(data, flt)

    print('Total error: {}'.format(error))


if __name__ == '__main__':
    main()
