#!/usr/bin/env python3
# coding: utf-8

import sys
import cv2
import numpy as np

spectrum = {'r': 0.2077, 'g': 0.895, 'b': 0.4264}

red = int(spectrum['r'] * 255)
green = int(spectrum['g'] * 255)
blue = int(spectrum['b'] * 255)

print(red, green, blue)

height = 400
width = 500
result = np.full((height, width, 3), (blue, green, red))
# result = np.full((height, width, 3), (0, 255, 0))
# result = np.full((height, width, 3), (blue, 0, red))

# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("result.png", result)


