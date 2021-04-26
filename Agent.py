# https://pillow.readthedocs.io/en/stable/reference/Image.html

import numpy
import math
from PIL import Image

class Agent:
    # Takes an image_path to prepare the image
    def __init__(self, image_path):
        self.image_path = image_path
        self.actual_image_left = None
        self.actual_image_right = None
        self.predict_image_right = None
        self.initialize_image()
        print("Loaded {}".format(image_path))
        return

    # Saves the current prediction to a given path
    def save(self, image_path):
        self.predict_image_right.save(image_path)
        print("Saved as {}".format(image_path))
        return

    # Convert the right half of the image to grayscale
    def initialize_image(self):
        with Image.open(self.image_path) as image:
            width, height = image.size
            middle = math.ceil(width/2)
            self.actual_image_left = image.crop( (0, 0, middle, height) )
            self.actual_image_right = image.crop( (middle, 0, width, height) )
            self.predict_image_right = self.actual_image_right.copy()
            data = list(self.predict_image_right.getdata())
        
        right_width = width - middle
        for y in range(0, height, 1):
            for x in range(0, right_width, 1):
                index = (right_width * y) + x
                grayscale = rgb_to_grayscale( data[index] )
                data[index] = (grayscale, grayscale, grayscale)
        self.predict_image_right.putdata(data)
        return

    # Get variance and gradient
    def get_variance(self):
        actual = numpy.array(self.actual_image_right.getdata())
        prediction = numpy.array(self.predict_image_right.getdata())
        gradient = prediction - actual
        variance = gradient.var()
        return (gradient, variance)

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
    save_all([agent.actual_image_left, agent.predict_image_right], "temp.png")

    data = agent.predict_image_right.getdata()

    # NOTE: .getpixel is slower than data[199]
    print(data[199]) # 199th pixel
    print(agent.actual_image_right.getpixel( (199, 0) )) # Pixel at x=199, y=0
    print()
    print(data[(200*399) + 5]) # (200*399) is the 1st column of the 400th row, +5 is the 6th column in that row
    print(agent.actual_image_right.getpixel( (5, 399) )) # Pixel at x=5, y=399
    print()
    print(agent.get_variance())