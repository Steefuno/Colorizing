import numpy
import scipy.stats
import multiprocessing
from functools import partial
import math
from PIL import Image
import Agent

from time import time

nclusters = 5 # Given, number of color clusters
change_convergence = 1 # Random small number decided for average change in cluster positions to declare convergence
patch_width = 3 # Width and height of patch to match testing data with training data
assert(patch_width%2 == 1)

class BasicAgent(Agent.Agent):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.test_data = list(self.test_image.getdata())
        self.training_data_gray = list(Agent.image_to_grayscale(self.training_image.copy()).getdata())
        self.training_patches = None
        self.get_training_data_patches()
        self.training_data_colors = None
        self.cluster_training_data()
        self.prediction = None
        return

    # For each pixel in the test data, get the 6 best grayscale match in the training data
    # Sets test data color to either majority cluster colors of the 6 matches, or the color of the best match if no majority
    def run(self):
        print("Predicting.")
        start_time = time()

        # daemonic processes are not allowed to have children, so we can't make processes here
        num_pixels = len(self.test_data)
        self.prediction = [None]*num_pixels
        for test_pixel_id in range(0, num_pixels, 1):
            self.prediction[test_pixel_id] = get_pixel_color(
                pixel_id=test_pixel_id,
                test_data=self.test_data,
                training_patches=self.training_patches,
                test_image_shape=self.test_image.size,
                training_data_colors=self.training_data_colors
            )
            print("Predicted Pixel {} of {} at t={:.4f}".format(test_pixel_id, num_pixels, time() - start_time))

        print("Predicted.")
        return self.prediction

    # Gets all patches from training_data_gray
    def get_training_data_patches(self):
        # For each pixel in the training data, get None or the patch as a (patch_width*patch_width, 3) array
        with multiprocessing.Pool() as pool:
            self.training_patches = pool.map(
                partial(
                    get_patch,
                    image_data=self.training_data_gray,
                    image_shape=self.training_image.size
                ),
                range(0, len(self.training_data_gray), 1)
            )
        print("Got training_patches.")
        return self.training_patches

    # Gets cluster colors and sets training_data_colors based 
    def cluster_training_data(self):
        cluster_colors = self.get_cluster_colors()
        image_data = self.training_image.getdata()
        print("Cluster Colors: {}".format(cluster_colors))

        # For each pixel, set to color of best cluster
        with multiprocessing.Pool() as pool:
            image_data = pool.map(
                partial(
                    get_pixel_cluster_id,
                    cluster_colors=cluster_colors,
                    get_cluster_color=True
                ),
                image_data
            )

        self.training_data_colors = image_data
        print("Loaded training_data_colors.")
        return image_data

    # Uses k-means to get nclusters cluster colors
    def get_cluster_colors(self):
        training_data = self.training_image.getdata() # Pixel data of the image
        width, height = self.training_image.size
        max_iterations = width * height

        # Assign starting centers
        cluster_colors = numpy.linspace(
            start = [0, 255 / (nclusters + 1) * 1, 0],
            stop = [0, 255 / (nclusters + 1) * nclusters, 0],
            num = nclusters,
            dtype = float,
            endpoint = True
        )
        
        # Repeat until convergence
        i = 0
        while i < max_iterations:
            # For each pixel, find cluster
            with multiprocessing.Pool() as pool:
                pixel_datas = pool.map(
                    partial(
                        get_pixel_cluster_id,
                        cluster_colors=cluster_colors,
                        get_cluster_color=False
                    ),
                    training_data
                )
            pixel_datas = numpy.array(pixel_datas, dtype=numpy.object_)

            # Get colors of each cluster
            new_cluster_colors = numpy.ndarray((nclusters, 3), dtype=float)
            for (cluster_id, cluster_color) in enumerate(cluster_colors):
                # Get color of pixels that matches the cluster_id
                pixel_data = pixel_datas[pixel_datas[:,0] == cluster_id][:,1]
                new_cluster_colors[cluster_id] = pixel_data.mean()
                
            # Check if change is small enough to converge
            change = numpy.sum(
                numpy.power(
                    numpy.subtract(cluster_colors, new_cluster_colors),
                    2
                )
            )
            cluster_colors = new_cluster_colors
            if change < change_convergence:
                break
            i += 1

        # Convert float cluster_colors to ints
        cluster_colors = numpy.floor(cluster_colors).astype(numpy.int16).tolist()
        return cluster_colors

# Find cluster for a single pixel
def get_pixel_cluster_id(pixel_data, cluster_colors, get_cluster_color=False):
    pixel_data = numpy.array(pixel_data)
    best_distance = numpy.sum(
        numpy.power(
            numpy.subtract(cluster_colors[0], pixel_data),
            2
        )
    ) # distance to center
    best_cluster_id = 0
    # For each cluster, find best
    for cluster_id in range(1, nclusters, 1):
        distance = numpy.sum(
            numpy.power(
                numpy.subtract(cluster_colors[cluster_id], pixel_data),
                2
            )
        )
        if distance < best_distance:
            best_cluster_id = cluster_id
            best_distance = distance
    # Gets either (best cluster id, pixel color) or (best cluster color)
    if get_cluster_color == False:
        return [best_cluster_id, pixel_data]
    else:
        return tuple(cluster_colors[best_cluster_id])

# Determined a color of a test_pixel based on the matching patches from training_data_gray
def get_pixel_color(pixel_id, test_data, training_patches, test_image_shape, training_data_colors):
    # Get the 6 best patches in training_data_gray that match the patch around pixel_id in the test_data
    best_patches = get_best_patches(
        pixel_id,
        test_data,
        training_patches,
        test_image_shape
    )
    # Pixel will be black if too close to the edge of the image to get a patch
    if best_patches is None:
        return (0,0,0)

    # Get colors for each best patch
    with multiprocessing.Pool() as pool:
        patch_colors = pool.map(
            partial(
                get_color,
                image_data=training_data_colors
            ),
            best_patches[:,1]
        )
    patch_colors = [None]*len(best_patches)
    
    # Find 2 most common colors
    mode = get_N_mode(patch_colors, 2)
    # If there is a single majority color, pixel is that color
    if mode[0] != mode[1]:
        return mode[0]
    # If there is no single majority color, pixel is color of center of best patch
    return training_data_colors[best_patches[0][1]]

# Gets a pixel's color in an image_data
def get_color(pixel_id, image_data):
    return image_data[pixel_id]

# Gets the N most common colors
def get_N_mode(data, N):
    counts = {}
    order = [None]*N
    for color in data:
        if color not in counts:
            counts[color] = 1
        else:
            counts[color] += 1

        for (index, top_color) in enumerate(order):
            if (top_color == None) or (counts[top_color] < counts[color]):
                order[index], color = color, order[index]
    return order

# Searches for the 6 most similar grayscale 3x3 patches in training_data_gray to the select 3x3 patch in test_data
def get_best_patches(pixel_id, test_data, training_patches, test_image_shape):
    test_patch = get_patch(pixel_id, test_data, test_image_shape)
    if test_patch is None:
        return None

    # Get L2 norm of test_patch compared to each patch in the training image
    with multiprocessing.Pool() as pool:
        norms = pool.map(
            partial(
                compare_patches,
                test_patch=test_patch,
                training_patches=training_patches
            ),
            range(0, len(training_patches), 1)
        )
    # Sort and get best 6
    norms = numpy.array(norms)
    norms = norms[norms[:,0] > 0]
    norms = norms[numpy.argsort(norms[:,0])][0:6]
    return numpy.flip(norms, axis=0)

# Gets the 3x3 patches around both pixels and gets the L2 Norm
def compare_patches(training_patch_id, test_patch, training_patches):
    training_patch = training_patches[training_patch_id]
    # Ignore if no patch
    if training_patch is None:
        return (-1, training_patch_id)

    norm = numpy.sum(
        numpy.power(
            numpy.subtract(test_patch, training_patch),
            2
        )
    )
    return [norm, training_patch_id]

# Gets the 3x3 patch around a pixel in an image
def get_patch(pixel_id, image_data, image_shape):
    width, height = image_shape
    padding = math.floor(patch_width/2)
    pixel_x = pixel_id % width
    pixel_y = math.floor(pixel_id / width)

    # If pixel is too close to left edge
    if (pixel_x < padding):
        return None
    # If pixel is too close to right edge
    if (pixel_x > width - padding - 1):
        return None
    # If pixel is too close to top edge
    if (pixel_y < padding):
        return None
    # If pixel is too close to bottom edge
    if (pixel_y > height - padding - 1):
        return None

    # Append to patch_data row by row
    patch_data = numpy.ndarray((0, 3), dtype=numpy.int16)
    for y in range(pixel_y - padding, pixel_y + padding + 1, 1):
        start = pixel_x - padding + (width * y)
        end = pixel_x + padding + 1 + (width * y)
        patch_data = numpy.append(patch_data, image_data[start:end])
    return patch_data

# Example usage of BasicAgent.py
if __name__ == "__main__":
    agent = BasicAgent("./image0.jpg")
    agent.run()

    image = Image.fromarray(agent.prediction)
    Agent.save_all([image], image_path)