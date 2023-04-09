import asyncio
import os
import time
from threading import Thread

import cv2
import gi
import argparse

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GLib


class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, source, fps_max, fps, show_stat, **properties):
        super(SensorFactory, self).__init__(**properties)

        self.capture = cv2.VideoCapture(source)
        self.status = False
        self.frame = None

        # self.image_width = int(self.capture.get(3))
        # self.image_height = int(self.capture.get(4))
        self.image_width = os.getenv('SOURCE_WIDTH', int(self.capture.get(3)))
        self.image_height = os.getenv('SOURCE_HEIGHT', int(self.capture.get(4)))

        self.fps_max = fps_max
        self.fps = fps

        self.number_frames = 0
        self.for_fps_frames_time = 0
        self.for_fps_frames_counter = 0
        self.current_fps = 0

        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in seconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast key-int-max=20 tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.image_width,
                                                                                     self.image_height, self.fps_max)

        self.last_time = 0
        self.frame_update = None
        self.show_stat = show_stat

        self.thread_stop = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        print('RTSP server is enabled')

    def set_fps(self, fps):
        self.thread_stop = True
        self.fps = fps
        self.duration = 1 / fps * Gst.SECOND

        self.number_frames = 0
        self.for_fps_frames_time = 0
        self.for_fps_frames_counter = 0
        self.current_fps = 0

        self.last_time = 0

        self.thread_stop = False

    def update(self):
        while not self.thread_stop:
            if self.capture.isOpened():
                start_time = time.time()

                status, frame = self.capture.read()

                if self.frame_update is not None:
                    frame = self.frame_update(frame)

                self.status = status
                self.frame = frame

                self.for_fps_frames_counter += 1
                self.for_fps_frames_time += (time.time() - self.last_time)
                if self.for_fps_frames_counter > (self.fps / 2):
                    self.current_fps = round(1 / (self.for_fps_frames_time / (self.fps / 2)))
                    self.for_fps_frames_time = 0
                    self.for_fps_frames_counter = 0

                self.last_time = time.time()

                if self.show_stat:
                    self.print_stat()

                end_time = time.time()
                if (end_time - start_time) < (self.duration / Gst.SECOND):
                    if (self.duration / Gst.SECOND - (end_time - start_time)) > 0:
                        time.sleep(self.duration / Gst.SECOND - (end_time - start_time))

    def on_need_data(self, src, length):
        if self.capture.isOpened():
            status = self.status
            frame = self.frame

            if status:
                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)

                if retval != Gst.FlowReturn.OK:
                    print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    def print_stat(self):
        os.system('clear')
        print("{}x{} {}FPS".format(self.image_width, self.image_height, self.current_fps))


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, source, fps_max, fps, port, uri, frame_update, show_stat, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(source, fps_max, fps, show_stat)
        self.factory.frame_update = frame_update
        self.factory.set_shared(True)
        self.set_service(str(port))
        self.get_mount_points().add_factory("/" + uri, self.factory)
        self.attach(None)


class RtspServer:
    def __init__(self, source=0, fps_max=120, fps=30, port=8554, uri="video", show_stat=False):
        Gst.init(None)

        self.server = GstServer(source, fps_max, fps, port, uri, self.frame_update, show_stat)
        self.loop = None

    def start(self):
        self.loop = GLib.MainLoop()
        self.loop.run()

    def frame_update(self, frame):
        return frame

    @property
    def current_fps(self):
        return self.server.factory.current_fps

    @property
    def fps_max(self):
        return self.server.factory.fps_max

    @property
    def target_fps(self):
        return self.server.factory.fps

    def set_target_fps(self, fps):
        self.server.factory.set_fps(fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="device id for the video device or video file location")
    parser.add_argument("--fps", default=30, help="fps of the camera", type=int)
    parser.add_argument("--fps_max", default=120, help="max fps of the camera", type=int)
    parser.add_argument("--port", default=8554, help="port to stream video", type=int)
    parser.add_argument("--stream_uri", default="video", help="rtsp video stream uri")
    parser.add_argument("--stat", default=False, help="Show stream statistics", action=argparse.BooleanOptionalAction)
    opt = parser.parse_args()

    server = RtspServer(opt.source, opt.fps_max, opt.fps, opt.port, opt.stream_uri, opt.stat)
    server.start()

