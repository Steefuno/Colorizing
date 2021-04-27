# https://pillow.readthedocs.io/en/stable/reference/Image.html

import numpy
import math
from PIL import Image

class Agent:
    # Takes an image_path to prepare the image
    def __init__(self, image_path):
        self.image_path = image_path
        self.training_image = None # Left image
        self.actual_image = None # Right image
        self.test_image = None # Grayscaled right image
        self.initialize_image()
        print("Loaded {}".format(image_path))
        return

    # Loads the image and prepares data for the agent
    def initialize_image(self):
        with Image.open(self.image_path) as image:
            width, height = image.size
            middle = math.ceil(width/2)
            self.training_image = image.crop( (0, 0, middle, height) )
            self.actual_image = image.crop( (middle, 0, width, height) )
            self.test_image = self.actual_image.copy()
            image_to_grayscale(self.test_image)
        return

# Converts an image to grayscale
def image_to_grayscale(image):
    width, height = image.size
    data = list(image.getdata())
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            index = (width * y) + x
            grayscale = rgb_to_grayscale( data[index] )
            data[index] = (grayscale, grayscale, grayscale)
    image.putdata(data)
    return

# Converts RGB to Grayscale
def rgb_to_grayscale(rgb_data):
    return min(
        math.floor(
            (0.21 * rgb_data[0]) + 
            (0.72 * rgb_data[1]) + 
            (0.07 * rgb_data[2])
        ),
        255
    )

# Get variance and gradient
def get_variance(actual_image, predict_image):
    actual_data = numpy.array(actual_image.getdata())
    prediction_data = numpy.array(predict_image.getdata())
    gradient = prediction_data - actual_data
    variance = gradient.var()
    return (gradient, variance)

# Saves the images attached by the sides
def save_all(images, image_path):
    total_size = [0, 0]
    for image in images:
        width, height = image.size
        total_size[0] += width
        if height > total_size[1]:
            total_size[1] = height
    total_size = tuple(total_size)
    
    result_image = Image.new("RGB", (total_size))
    x = 0
    for image in images:
        width, height = image.size
        result_image.paste(image, (x, 0))
        x += width
    result_image.save(image_path)
    print("Saved as {}".format(image_path))
    return

# Example usage of Agent.py
if __name__ == "__main__":
    agent = Agent("./image0.jpg")
    save_all([agent.training_image, agent.test_image], "./Output/temp.png")

    data = agent.test_image.getdata()

    # NOTE: .getpixel is slower than data[199]
    print(data[199]) # 199th pixel
    print(agent.actual_image.getpixel( (199, 0) )) # Pixel at x=199, y=0
    print()
    print(data[(200*399) + 5]) # (200*399) is the 1st column of the 400th row, +5 is the 6th column in that row
    print(agent.actual_image.getpixel( (5, 399) )) # Pixel at x=5, y=399
    print()
    print(get_variance(agent.actual_image, agent.test_image))