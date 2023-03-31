import os
import time
import datetime
from threading import Thread

import cv2


class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)

        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        self.codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        self.output_video = cv2.VideoWriter('media/output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        self.last_time = 0
        self.updates = 0
        self.frames = 0

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                self.print_stat()

    def show_frame(self):
        if self.status:
            frame = self.frame

            cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        self.output_video.write(self.frame)

    def print_stat(self):
        os.system('clear')
        fps = round(1 / (time.time() - self.last_time))
        print("{}x{} {}FPS".format(self.frame_width, self.frame_height, fps))
        self.last_time = time.time()


if __name__ == '__main__':
    rtsp_stream_link = "rtsp://127.0.0.1:8555/video_stream"
    # rtsp_stream_link = 0
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link)

    while True:
        try:
            video_stream_widget.show_frame()
            # video_stream_widget.save_frame()
            pass
        except AttributeError:
            pass
