import time
from copy import deepcopy
from functools import lru_cache

#from doctr.utils.visualization import draw_boxes
import cv2
import matplotlib.pyplot as plt

from doctr.io import DocumentFile
from doctr.models import detection_predictor

start_time = time.time()

# Detect
@lru_cache
def get_model_doctr(arch='db_resnet50'):
    model = detection_predictor(
        arch='db_resnet50', pretrained=True, assume_straight_pages=True)
    return model

def detection_doctr(image, model):

    single_img_doc = DocumentFile.from_images(image)
    result = model(single_img_doc)

    h, w ,c = single_img_doc[0].shape
    bboxes = []
    for box in result[0]:
        x1 = int(box[0]*w)
        y1 = int(box[1]*h)
        x2 = int(box[2]*w)
        y2 = int(box[3]*h)
        bboxes.insert(0, [x1, y1, x2, y2])

    return bboxes
detect_model = get_model_doctr()
boxes= detection_doctr("./img_test/mcocr_public_145013cgpey.jpg", detect_model)
bboxes = deepcopy(boxes)

img =cv2.imread('./img_test/mcocr_public_145013cgpey.jpg')
imageRectangle = img.copy()
#h, w = img.shape[:2]
# define the starting and end points of the rectangle
for box in bboxes:#.tolist():
    xmin, ymin, xmax, ymax = box
    image = cv2.rectangle(
            imageRectangle,
            (xmin, ymin),
            (xmax, ymax),
            color= (0, 0, 255),
            thickness=2)

cv2.imshow('display image', image)
cv2.waitKey(0)
print("--- %s seconds ---" % (time.time() - start_time))
