import asyncio

from sanic import Sanic

from web.routes import setup_routes
from web.plugins import middlewares
from sanic_cors import CORS

from gi.repository import Gst, GstRtspServer, GLib


class WebServer:
    def __init__(self, app, run_test=False):
        super().__init__()
        self.sanic = Sanic(name='SmartEye')

        self.loop = app.loop
        self.__app = app

        CORS(self.sanic,
             resources={r"/api/*": {"origins": "*"}, r"/auth/*": {"origins": "*"}},
             automatic_options=True
             )

        self.sanic.update_config({
            "ACCESS_LOG": True,
            "FALLBACK_ERROR_FORMAT": "json",
            "KEEP_ALIVE": True,
            "KEEP_ALIVE_TIMEOUT": 600})

        middlewares.setup_middlewares(self.sanic)

        setup_routes(self, app)

    def start(self):
        self._init_web_server()

    def _init_web_server(self):
        self.server = self.sanic.create_server(
            host='0.0.0.0',
            port=9999,
            return_asyncio_server=True,
            debug=False
        )

        self.loop.create_task(self.server)
