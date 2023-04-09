import os

import dotenv

from neural_net.NeuralNet import NeuralNetwork
import argparse
import cv2
import numpy as np
import datetime

from RtspServer import RtspServer


class RtspMergeServer(RtspServer):
    def __init__(self, source=0, fps_max=120, fps=None, port=8554, uri="video", show_stat=False, show_osd=False, frame_skip=0):
        self.model = NeuralNetwork()

        self.show_osd = show_osd
        self.neural_osd = None
        self.frame_skip = frame_skip
        self.frame_counter = 0
        self.active_class = dict()
        self._activity_log = []
        self._activity_log_last_time = 0

        self.supported_class = {
            "person": True,
            "car": True,
            "bicycle": True,
            "motorbike": True,
            "bus": True,
            "truck": True,
            "boat": True,
            "backpack": True,
            "handbag": True,
            "cell phone": True,
        }

        super().__init__(source=source,
                         fps_max=fps_max,
                         fps=fps if fps is not None else fps_max,
                         port=port,
                         uri=uri,
                         show_stat=show_stat)

    def frame_update(self, frame):
        frame = cv2.putText(frame, datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not self.show_osd:
            return frame

        if self.frame_counter > self.frame_skip:
            self.frame_counter = 0

        if self.frame_counter == 0:
            self.neural_osd = self.model.detect_objects(frame)

        self.frame_counter += 1

        frame = self.draw_outputs(frame, self.neural_osd, self.model.class_names, self.whitelist)

        return frame

    def draw_outputs(self, frame, outputs, class_names, white_list=None):
        boxes, score, classes, nums = outputs
        boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]

        wh = np.flip(frame.shape[0:2])

        active_class = dict()
        for i in range(nums):
            cl = class_names[int(classes[i])]

            if white_list is not None and cl not in white_list:
                continue

            if cl not in active_class:
                active_class[cl] = 1
            else:
                active_class[cl] += 1

            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

            if x1y1[0] < 0:
                x1y1 = (0, x1y1[1])
            if x1y1[1] < 0:
                x1y1 = (x1y1[0], 0)

            text_pos = (x1y1[0], x1y1[1] + 16)

            frame = cv2.rectangle(frame, x1y1, x2y2, (255, 0, 0), 2)
            frame = cv2.putText(frame, '{} {:.2f}'.format(
                class_names[int(classes[i])].capitalize(), score[i]),
                                text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        self.active_class = active_class
        self.update_activity_log(active_class)

        return frame

    def update_activity_log(self, active_class):
        fps_ms = 1 / int(os.getenv('ACTIVITY_LOG_FPS'))
        time = datetime.datetime.now()

        if (time.timestamp() - self._activity_log_last_time) < fps_ms:
            return

        self._activity_log_last_time = time.timestamp()
        res = 0
        for key in active_class:
            res += active_class[key]

        if len(self._activity_log) >= int(os.getenv('ACTIVITY_LOG_LENGTH')):
            self._activity_log.pop(0)

        self._activity_log.append(res)

    @property
    def activity_log(self):
        return self._activity_log

    @property
    def whitelist(self):
        return [cl for cl in self.supported_class if self.supported_class[cl]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="device id for the video device or video file location")
    parser.add_argument("--fps", default=120, help="fps of the camera", type=int)
    parser.add_argument("--fps_max", default=120, help="max fps of the camera", type=int)
    parser.add_argument("--port", default=8554, help="port to stream video", type=int)
    parser.add_argument("--stream_uri", default="video", help="rtsp video stream uri", type=str)
    parser.add_argument("--stat", default=False, help="Show stream statistics", action=argparse.BooleanOptionalAction)
    parser.add_argument("--osd", default=False, help="Show neural osd", action=argparse.BooleanOptionalAction)
    parser.add_argument("--frame_skip", default=0, help="Skip frames for neural network processing", type=int)
    opt = parser.parse_args()

    server = RtspMergeServer(opt.source, opt.max_fps, opt.fps, opt.port, opt.stream_uri, opt.stat, opt.osd, opt.frame_skip)
    server.start()
