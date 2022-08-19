import numpy as np
import json
import torch
from doctr.io import DocumentFile
from doctr.models import detection_predictor
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg 
from functools import lru_cache
import time


start_time = time.time()

# Detect
@lru_cache
def get_model_doctr(arch='db_resnet50'):
    model = detection_predictor(
        arch='db_resnet50', pretrained='./db_resnet50.pt', assume_straight_pages=True)
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

    return bboxes, single_img_doc[0], h, w


# Sắp xếp box theo thứ tự trái -> phải, trên -> dưới
# g=arrange_bbox(df["img_bboxes"][i])
# rows=arrange_row(g=g)

def arrange_bbox(bboxes):
    n = len(bboxes)
    xcentres = [(b[0] + b[2]) // 2 for b in bboxes]
    ycentres = [(b[1] + b[3]) // 2 for b in bboxes]
    heights = [abs(b[1] - b[3]) for b in bboxes]
    width = [abs(b[2] - b[0]) for b in bboxes]

    def is_top_to(i, j):
        result = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 3)
        return result

    def is_left_to(i, j):
        return (xcentres[i] - xcentres[j]) > ((width[i] + width[j]) / 3)

    # <L-R><T-B>
    # +1: Left/Top
    # -1: Right/Bottom
    g = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(n):
            if is_left_to(i, j):
                g[i, j] += 10
            if is_left_to(j, i):
                g[i, j] -= 10
            if is_top_to(i, j):
                g[i, j] += 1
            if is_top_to(j, i):
                g[i, j] -= 1
    return g


def arrange_row(bboxes=None, g=None, i=None, visited=None):
    if visited is not None and i in visited:
        return []
    if g is None:
        g = arrange_bbox(bboxes)
    if i is None:
        visited = []
        rows = []
        for i in range(g.shape[0]):
            if i not in visited:
                indices = arrange_row(g=g, i=i, visited=visited)
                visited.extend(indices)
                rows.append(indices)
        return rows
    else:
        indices = [j for j in range(g.shape[0]) if j not in visited]
        indices = [j for j in indices if abs(g[i, j]) == 10 or i == j]
        indices = np.array(indices)
        g_ = g[np.ix_(indices, indices)]
        order = np.argsort(np.sum(g_, axis=1))
        indices = indices[order].tolist()
        indices = [int(i) for i in indices]
        return indices

def split_row(rows,bboxes,w,ratio):
    xcentres = [(b[0] + b[2]) // 2 for b in bboxes]
    x1x2= [ [b[0],b[2]] for b in bboxes]  
    mean_hight=np.mean( [abs(b[1] - b[3]) for b in bboxes]) 
    new_rows=[]

    print("mean_hight: ",mean_hight)
    max_width= int(ratio*mean_hight)
    for row in rows:
        new_row=[row[0]]
        for i in range(1,len(row)):
            if abs(x1x2[row[i]][0]-x1x2[row[i-1]][1]) > max_width:
                new_rows.append(new_row)
                new_row=[row[i]]
            else:
                new_row.append(row[i])
        new_rows.append(new_row)
    
    return new_rows

# def convert_box_to_XYXY(bboxes,type_in="XXYY"):
#             new_bboxes=[]
#             for b in bboxes:
#                 new_bboxes.append([b[0],b[2],b[1],b[3]])
#             return new_bboxes

#Merge box,text, get map
def get_mapping(box_to_merge,text_to_merge):
        mapping=[]
        merged_boxes=[]
        merged_texts=[]
        def merge_box(bboxes):
            min_x=min([b[0] for b in bboxes])
            max_x=min([b[2] for b in bboxes])
            min_y=min([b[1] for b in bboxes])
            max_y=min([b[3] for b in bboxes])
            return [min_x,min_y,max_x,max_y]


        for i in range(len(box_to_merge)):
            merged_box=merge_box(box_to_merge[i])
            merged_text=""
            for text in text_to_merge[i]:
                merged_text+=" "+text
            merged_boxes.append(merged_box)
            merged_texts.append(merged_text)
            for j in range(len(box_to_merge[i])):
                mapping.append(i)
        return mapping, merged_boxes,merged_texts
        
        
# Recognition
# input box: x1,y1,x2,y2
@lru_cache
def get_model_vietocr():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    # config['weights'] = 'https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4'
    #config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA' #(transformer)
    config['weights'] = './vgg19_bn_seq2seq.pth'
    config['cnn']['pretrained'] = False
    if torch.cuda.is_available():
        config['device'] = 'cuda:0'
    else:
        config['device'] = 'cpu'

    config['predictor']['beamsearch'] = False
    model = Predictor(config)
    return model


def recognition_vietocr(image, bboxes, model):
    raw_text = []
    # image = np.frombuffer(image, np.uint8)
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    for box in bboxes:
        # print("image.shape: ",image.shape)
        # print("box: ",box)
        img_box = image[box[1]:box[3], box[0]:box[2]]
        # print("img_box.shape: ",img_box.shape)
        img_box = Image.fromarray(img_box)
        text = model.predict(img_box)
        if text == []:
            raw_text.append("?")
            continue
        raw_text.append(str(text))
    return raw_text


detect_model = get_model_doctr()
bboxes, image, h, w = detection_doctr("./test/Payment 08.03.22_page_13.jpg", detect_model)

recognize_model = get_model_vietocr()
raw_text = recognition_vietocr(image, bboxes, recognize_model)

g = arrange_bbox(bboxes)
rows = arrange_row(g=g)
rows = split_row(rows,bboxes,w,ratio =0.8)

new_text = []
new_box = []
box_to_merge=[]
text_to_merge=[]
for i in range(len(rows)):
    box_row=[]
    text_row=[]
    for j in rows[i]:
        new_text.append(raw_text[j])
        new_box.append(bboxes[j])
        text_row.append(raw_text[j])
        box_row.append(bboxes[j])
    box_to_merge.append(box_row)
    text_to_merge.append(text_row)
    
mapping, merged_boxes, merged_texts=get_mapping(box_to_merge,text_to_merge)

s=dict( texts=new_text,
            bboxes=new_box,
            height=h,
            width=w,
            mapping=mapping,
            merged_boxes=merged_boxes,
            merged_texts=merged_texts,
        )

#print(s)
with open("saved.json", "w", encoding="utf-8") as f:
    json.dump(s, f, indent=2,ensure_ascii=False)
print("--- %s seconds ---" % (time.time() - start_time))
