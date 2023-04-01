from RtspMergeServer import RtspMergeServer
from web import WebServer
import asyncio
import uvloop


class App:
    def __init__(self):
        self.loop = None
        self.web = None
        self.RtspServer = None

    def initialize(self):
        self.loop = asyncio.get_event_loop()
        self.init_web()
        self.init_rtsp()

    def init_web(self):
        self.web = WebServer(self)

    def init_rtsp(self):
        self.RtspServer = RtspMergeServer(self)

    def run(self):
        self.initialize()
        self.RtspServer.start()
        self.web.start()

        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            print('console', e)
            self.stop()

    def stop(self):
        self.loop.stop()


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
