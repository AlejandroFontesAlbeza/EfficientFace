import torch

import cv2
import imageio
import time
from utils import MyModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = MyModel().to(device)

model.load_state_dict(torch.load("../resources/efficientNetFace.pth", map_location=device))
model.eval()



def inference(video_path, output_gif_path) -> None:

    cap = cv2.VideoCapture(video_path)
    frames_for_gif = []

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

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames_for_gif.append(frame_rgb)        
        cv2.imshow("Keypoints", frame_resized)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

    imageio.mimsave(output_gif_path, frames_for_gif, fps = 30)
    print("GIF saved")

if __name__ == "__main__":

    video_path = "../resources/inputs/selfievideo2.mp4"
    output_gif_path = "../resources/inputs/selfievideo2.gif"
    inference(video_path=video_path, output_gif_path=output_gif_path)