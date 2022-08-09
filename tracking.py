import matplotlib.pyplot as plt
import argparse
import sys
import yaml
import os
import torch
import cv2 as cv
from deep_sort.deep_sort import DeepSort
CONFIG_PATH="./config/config.yaml"

def load_config(config_name):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    return config

def run(config=CONFIG_PATH, weight=None, source:str=0, display=True, save=True):
    '''
    Paramerters:
        config: Path to config file, default .\config\config.yaml
        weight: Pretrained checkpoint. if None will use checkpoint in config file. Otherwise download from torchhub
        source: Open correspond video path to process, number (0,1,2) for webcam
        display: Display video to screen
        save: Path to save video after processing completely
    '''
    # Load config
    config = load_config(config)

    ### Detection ###
    if weight is None:
        model = torch.hub.load('yolov5', 'custom', path=config['yolo_detector']['weight'], source='local', _verbose=False)  # or yolov5n - yolov5x6, custom
    else:
        model = torch.hub.load('ultralytics/yolov5', weight)  

    # Config Inference
    model.classes = config["yolo_detector"]["classes"]
    model.conf = config["yolo_detector"]["conf"]
    model.iou = config["yolo_detector"]["iou"]
    
    ### Tracker ###
    deepsort = DeepSort(model_path=config['deepsort_tracker']['model_path'],
                max_dist=config['deepsort_tracker']['max_dist'],
                min_confidence=config['deepsort_tracker']['min_confidence'], 
                nms_max_overlap=config['deepsort_tracker']['nms_max_overlap'],
                max_iou_distance=config['deepsort_tracker']['max_iou_distance'], 
                max_age=config['deepsort_tracker']['max_age'], 
                n_init=config['deepsort_tracker']['n_init'], 
                nn_budget=config['deepsort_tracker']['nn_budget'], 
                use_cuda=config['deepsort_tracker']['use_cuda'])

    ### Video ###
    if isinstance(source, str):
        if source.isnumeric():
            source = int(source)
    vid = cv.VideoCapture(source)

    # Check whether if video is opened successfully
    if (vid.isOpened() == False):
        print("Error opening video stream/file")
    else:
        print("Streaming.....")

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    size = (int(frame_width * config["video"]['scale_video_size']), 
                int(frame_height * config["video"]['scale_video_size']))

    record=None
    if save:
        if source.isnumeric():
            video_name = source
        else: 
            video_name = os.path.basename(source)
            video_name = video_name.split('.')[0]
        os.makedirs("result")
        record = cv.VideoWriter(f'result/{video_name}.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         30, size)
    if display:
        print("-" * 27)
        print("| Press 'Q' to quit video |")
        print("-" * 27)

    while(vid.isOpened()):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if ret is True:
            # Reshape
            frame = cv.resize(frame, size, interpolation = cv.INTER_AREA)
            frame_temp = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
            # Inference
            predictions = model(frame_temp)

            # Results
            result = predictions.xywh[0]
            # bboxxywhs = bboxxywhs.astype(int)

            track_result = deepsort.update(result[:,:4].numpy(), result[:, -2].numpy(), frame)
            if len(track_result) > 0:
                # print(f"Track result: \n{track_result}")
                # track_boudingbox = deepsort._xywh_to_xyxy(track_result[:, :4])
                track_box, track_id = track_result[:, :4], track_result[:, -1]

                for i in range(len(track_result)):
                    x1, y1, x2, y2 = track_box[i, 0], track_box[i, 1], track_box[i, 2], track_box[i, 3]
                    frame = cv.rectangle(frame, (x1, y1), (x2, y2),
                                        tuple(config["draw"]['bouding_box']['color']),
                                        config["draw"]['bouding_box']['thick'])
                    frame = cv.putText(frame, str(track_id[i]), (int(x1) - 8, int(y1) - 4),
                                        cv.FONT_HERSHEY_SIMPLEX, 
                                        config["draw"]['text']['font_scale'], 
                                        tuple(config["draw"]['text']['color']), 
                                        config["draw"]['text']['thick'])

                # Save video
                if record is not None:
                    record.write(frame)

                # Display the resulting frame
                if display:
                    cv.imshow('Tracking', frame)
                    # press 'q' to quit
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        print("Finish!")
                        break
        else:
            print("Finish!")
            break    

    vid.release()
    if record is not None:
        record.release()
    cv.destroyAllWindows()

def parse_opt():
    '''
    Paramerters:
        config: Path to config file, default .\config\config.yaml
        weight: Pretrained checkpoint. if None will use checkpoint in config file. Otherwise download from torchhub
        source: Open correspond video path to process, number (0,1,2) for webcam
        display: Display video to screen
        save: Path to save video after processing completely
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help="add config file")
    parser.add_argument('--weight', type=str, default=None, help="YoloV5 checkpoint. By default take checkpoint path in config. Otherwise will load from tochhub")
    parser.add_argument('--source', type=str, default=0, help="Input to process. Default is '0' for webcam. Otherwise give relative path to video")
    parser.add_argument('--display', action=argparse.BooleanOptionalAction, help="Show video")
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help="Save video")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)