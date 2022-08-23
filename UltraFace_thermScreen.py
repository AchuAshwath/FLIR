"""
This code uses the pytorch model to detect faces from live video or camera.
"""
import argparse
import sys
import cv2
import numpy as np
import os

from vision.ssd.config.fd_config import define_img_size

count = 0
flag = None

def pixel_to_temperature(pixel):
    temp_min = 80
    temp_max = 110
    pixel_max = 255
    pixel_min = 0
    temp_range = temp_max-temp_min
    temp = (((pixel-pixel_min)*temp_range)/(pixel_max-pixel_min))+temp_min + 4.3
    return temp

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3264,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=480, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--video_path', default="/home/srecflir/thermal-surveillance/crowd_Trim.mp4", type=str,
                    help='path of video')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

net_type = args.net_type

#cap = cv2.VideoCapture(args.video_path)  # capture from video
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) # capture from camera
#cap = cv2.VideoCapture(1)


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

timer = Timer()
sum = 0
i = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("end")
        break
    output = orig_image
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    faces, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    interval = timer.end()
    if faces ==[]:
        face = False
        flag = 0
    else: 
        face = True
    
    for (x1,y1,x2,y2) in faces:
        if len(faces)>1:
            count+=1
        roi = output[int(y1):int(y2), int(x1):int(x2)]
        try:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            continue

        try:
            if face == True:
                if flag ==0:
                    count+=len(faces)
                    flag = 1
                        

        except ValueError:
            continue
        

        # Mask is boolean type of matrix.
        mask = np.zeros_like(roi_gray)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = pixel_to_temperature(np.mean(roi_gray))

        # Colors for rectangles and textmin_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < 100 else (0, 0, 255)
        

        # Draw rectangles for visualisation
        output = cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.putText(output, "{} F".format(temperature), (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)  
        if temperature > 100:
            try:
                os.mkdir('persons/person'+str(count))
            except FileExistsError:
                pass
            while(face is True):
                face = output[int(y1)+2:int(y2)-1,int(x1)+2:int(x2)-1]
                cv2.imwrite('persons/person'+str(count)+'/person'+str(count)+'_face'+str(i)+'.jpg', face)
                print("image captured")
                i+=1

    cv2.imshow('Thermal', output)    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("all face num:{}".format(sum))
