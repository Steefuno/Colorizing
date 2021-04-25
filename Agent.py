# https://pillow.readthedocs.io/en/stable/reference/Image.html

import numpy
import math
from PIL import Image

class Agent:
    # Takes an image_path to prepare the image
    def __init__(self, image_path):
        self.image_path = image_path
        self.initialize_image()
        print("Loaded {}".format(image_path))
        return

    # Saves the current image to a given path
    def save(self, image_path):
        self.image.save(image_path)
        print("Saved as {}".format(image_path))
        return

    # Convert the right half of the image to grayscale
    def initialize_image(self):
        with Image.open(self.image_path) as im:
            self.image = im
            data = list(self.image.getdata())
        width, height = self.image.size
        
        for y in range(0, height, 1):
            for x in range(math.ceil(width/2), width, 1):
                index = (height * y) + x
                grayscale = rgb_to_grayscale( data[index] )
                data[index] = grayscale
        self.image.putdata(data)
        self.data = data
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

# Example usage of Agent.py
if __name__ == "__main__":
    agent = Agent("./image0.jpg")
    agent.save("temp.jpg")

    data = agent.image.getdata()
    print(data[399])
    print(agent.image.getpixel( (399, 0) ))
    print(data[(400*399) + 0]) # (399*399) is the 400th row, +0 is the 0th column in that row
    print(agent.image.getpixel( (0, 399) ))