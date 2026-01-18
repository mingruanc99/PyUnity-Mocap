import cv2
import numpy as np

rmat = np.eye(3)
ret = cv2.RQDecomp3x3(rmat)
print(f"Number of return values: {len(ret)}")
print(f"Return values: {ret}")
