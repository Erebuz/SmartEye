from neural_net.NeuralNet import NeuralNetwork
import argparse
import cv2
import numpy as np
import datetime

from RtspServer import RtspServer


class RtspMergeServer(RtspServer):
    def __init__(self, app, source=0, fps=30, port=8554, uri="video", show_stat=False):
        self.model = NeuralNetwork()

        super().__init__(app,
                         source=source,
                         fps=fps,
                         port=port,
                         uri=uri,
                         show_stat=show_stat)

    def frame_update(self, frame):
        boxes, scores, classes, nums = self.model.detect_objects(frame)
        frame = self.draw_outputs(frame, (boxes, scores, classes, nums), self.model.class_names)

        frame = cv2.putText(frame, datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        return frame

    @staticmethod
    def draw_outputs(frame, outputs, class_names, white_list=None):
        boxes, score, classes, nums = outputs
        boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]
        wh = np.flip(frame.shape[0:2])
        for i in range(nums):
            if white_list is not None and class_names[int(classes[i])] not in white_list:
                continue
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            frame = cv2.rectangle(frame, x1y1, x2y2, (255, 0, 0), 2)
            frame = cv2.putText(frame, '{} {:.4f}'.format(
                class_names[int(classes[i])], score[i]),
                                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="device id for the video device or video file location")
    parser.add_argument("--fps", default=30, help="fps of the camera", type=int)
    parser.add_argument("--port", default=8555, help="port to stream video", type=int)
    parser.add_argument("--stream_uri", default="video_stream", help="rtsp video stream uri")
    parser.add_argument("--stat", default=False, help="Show stream statistics")
    opt = parser.parse_args()

    RtspMergeServer(opt.source, opt.fps, opt.port, opt.stream_uri, opt.stat)
