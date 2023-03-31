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
    def __init__(self, source, fps, show_stat, **properties):
        super(SensorFactory, self).__init__(**properties)

        self.capture = cv2.VideoCapture(source)
        self.status = False
        self.frame = None

        self.image_width = int(self.capture.get(3))
        self.image_height = int(self.capture.get(4))

        self.number_frames = 0
        self.fps = fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.image_width,
                                                                                     self.image_height, self.fps)

        self.last_time = 0
        self.frame_update = None
        self.show_stat = show_stat

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        print('RTSP server is enabled')

    def update(self):
        while True:
            if self.capture.isOpened():
                status, frame = self.capture.read()

                if self.frame_update is not None:
                    frame = self.frame_update(frame)

                self.status = status
                self.frame = frame

                if self.show_stat:
                    self.print_stat()

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
        fps = round(1 / (time.time() - self.last_time))
        print("{}x{} {}FPS".format(self.image_width, self.image_height, fps))
        self.last_time = time.time()


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, source, fps, port, uri, frame_update, show_stat, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(source, fps, show_stat)
        self.factory.frame_update = frame_update
        self.factory.set_shared(True)
        self.set_service(str(port))
        self.get_mount_points().add_factory("/" + uri, self.factory)
        self.attach(None)


class RtspServer:
    def __init__(self, source=0, fps=30, port=8554, uri="video_stream", show_stat=False):
        Gst.init(None)

        self.server = GstServer(source, fps, port, uri, self.frame_update, show_stat)

        loop = GLib.MainLoop()
        loop.run()

    def frame_update(self, frame):
        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="device id for the video device or video file location")
    parser.add_argument("--fps", default=30, help="fps of the camera", type=int)
    parser.add_argument("--port", default=8554, help="port to stream video", type=int)
    parser.add_argument("--stream_uri", default="video_stream", help="rtsp video stream uri")
    parser.add_argument("--stat", default=False, help="Show stream statistics")
    opt = parser.parse_args()

    RtspServer(opt.source, opt.fps, opt.port, opt.stream_uri, opt.stat)
