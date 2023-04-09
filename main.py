import os.path
from datetime import datetime
from json import dumps

from RtspMergeServer import RtspMergeServer
from web import WebServer
import asyncio
import uvloop
from threading import Thread
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


class App:
    def __init__(self):
        self.loop = None
        self.web = None
        self.RtspMergeServer = None
        self.rtsp_thread = None
        self.root = os.path.dirname(os.path.abspath(__file__))

    def initialize(self):
        self.loop = asyncio.get_event_loop()
        self.init_web()

    def init_web(self):
        self.web = WebServer(self)

    def init_rtsp(self):
        self.RtspMergeServer = RtspMergeServer(
            show_osd=False,
            source=os.getenv('SOURCE', 0),
            fps_max=120,
            fps=30
        )

        self.rtsp_thread = Thread(target=self.RtspMergeServer.start, args=())
        self.rtsp_thread.daemon = True
        self.rtsp_thread.start()

    async def send_sockets(self):
        while True:
            await asyncio.sleep(0.2)
            await self.web.send_all_sockets({
                'action': 'update',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    "fps": {
                        'current': self.RtspMergeServer.current_fps,
                        'max': self.RtspMergeServer.fps_max,
                        'target': self.RtspMergeServer.target_fps,
                    },
                    "classes": self.RtspMergeServer.active_class
                }
            })

    def init_ws_update(self):
        self.loop.create_task(self.send_sockets())

    def run(self):
        self.initialize()
        self.web.start()
        self.init_rtsp()
        self.init_ws_update()

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
