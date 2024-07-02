import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(
    name="buffalo_l",
    root="/home/jeans/internship/face-vit/results",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread(
    "/home/jeans/internship/resources/datasets/lfw/Barry_Zito/Barry_Zito_0002.jpg"
)
faces = app.get(img)
breakpoint()
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
