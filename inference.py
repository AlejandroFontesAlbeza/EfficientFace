import torch
import torch.nn as nn
from torchvision import models

import cv2
import numpy as np
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyModel(nn.Module):

    def __init__(self, num_kpts = 30, grad_from = 2):
        super(MyModel,self).__init__()
        self.effnet = models.efficientnet_b0(pretrained = True)
        self.outputs_last_layer = 1280*3*3


        for name, param in self.effnet.features.named_parameters():
            if int(name.split('.')[0]) < grad_from:
                param.requires_grad = False


        self.regressor = nn.Sequential(nn.Dropout(p=0.6),nn.Linear(self.outputs_last_layer, num_kpts))


    def forward(self,x):
        x = self.effnet.features(x)
        x = torch.flatten(x,1)
        x = self.regressor(x)

        return x


model = MyModel().to(device)

model.load_state_dict(torch.load("eficientNetFace.pth", map_location=device))
model.eval()

video_path = "inputs/selfievideo.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame,(96,96))
    
    
    tensor_frame = torch.from_numpy(frame_resized).to(device) 

    tensor_frame = tensor_frame.permute(2, 0, 1).float() / 255.0 
    tensor_frame = tensor_frame.unsqueeze(0)

    # # Inferencia
    with torch.no_grad():
        start = time.time()
        preds = model(tensor_frame)
        end = time.time()
        print(end-start)

    kpts = preds.cpu().numpy().reshape(-1, 2)

    # Dibujar keypoints en frame_resized
    for x, y in kpts:
        x = int(x) 
        y = int(y)
        cv2.circle(frame_resized, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Keypoints", frame_resized)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()