import torch
import onnxruntime as onnxrt
import cv2
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
cfg= cfg_re50
device = "cuda:0"
img_path="/home/ds1/cuong2203/Pytorch_Retinaface/face.jpg"
img=cv2.imread(img_path)
#preprocess image
img = np.float32(img)
img=cv2.resize(img,(640,640))
im_height, im_width, _ = img.shape
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0)
img = img.to(device)
scale = scale.to(device)
onnx_session = onnxrt.InferenceSession("/home/ds1/cuong2203/Pytorch_Retinaface/FaceDetector.onnx")
onnx_input = {onnx_session.get_inputs()[0].name:img.cpu().detach().numpy()}
loc, conf, landms= onnx_session.run(None,onnx_input)
conf=torch.as_tensor(conf)
#print(loc.shape)
print(type(loc))
loc=torch.as_tensor(loc).reshape(1,16800,4).cuda()

print("LOC SHAPE :",loc.shape)

landms=torch.as_tensor(landms).reshape(1,16800,10).cuda()
print("lamds shape :",landms.shape)
priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
priors = priorbox.forward()
priors = priors.to(device)
prior_data = priors.data
#print(priors.shape)
resize=1
boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
boxes = boxes * scale / resize
boxes = boxes.cpu().numpy()
scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
scale1 = scale1.to(device)
landms = landms * scale1 / resize
landms = landms.cpu().numpy()

# ignore low scores
inds = np.where(scores > 0.4)[0]
boxes = boxes[inds]
landms = landms[inds]
scores = scores[inds]

# keep top-K before NMS
order = scores.argsort()[::-1][:5000]
#print(order)
boxes = boxes[order]
landms = landms[order]
scores = scores[order]

# do NMS
dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
keep = py_cpu_nms(dets, 0.4)
# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
dets = dets[keep, :]
landms = landms[keep]

# keep top-K faster NMS
dets = dets[:5000, :]
landms = landms[:5000, :]
save_image=True
dets = np.concatenate((dets, landms), axis=1)
print("dets shape ",dets.shape)
if save_image:
            for b in dets:
                if b[4] < 0.6:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                print(b)
                x,y,w,h=int(b[0]),int(b[1]),int(b[2]),int(b[3])
                cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
                cx = x
                cy = y + 12
                cv2.putText(img, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test123.jpg"
            cv2.imwrite(name, img)


