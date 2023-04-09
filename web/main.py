import json

from sanic import Sanic
from websockets.exceptions import ConnectionClosed

from web.sys_routes import load_sys_routes
from web.api_routes import load_api_routes
from web.plugins import middlewares
from sanic_cors import CORS

from sanic_jwt import Initialize
from web.auth import Auth


class WebServer:
    def __init__(self, app):
        super().__init__()
        self.sanic = Sanic(name='SmartEye')

        self.loop = app.loop
        self.__app = app
        self.sockets = []

        CORS(self.sanic,
             resources={r"/api/*": {"origins": "*"}, r"/auth/*": {"origins": "*"}},
             automatic_options=True,
             )

        self.sanic.update_config({
            "ACCESS_LOG": True,
            "FALLBACK_ERROR_FORMAT": "json",
            "KEEP_ALIVE": True,
            "KEEP_ALIVE_TIMEOUT": 600})

        middlewares.setup_middlewares(self.sanic)

        load_sys_routes(self, app)
        load_api_routes(self, app)

        self.auth = Auth(self)

    def start(self):
        self._init_web_server()

    def _init_web_server(self):
        Initialize(self.sanic,
                   expiration_delta=1800,
                   authenticate=self.auth.authenticate,
                   refresh_token_enabled=False,
                   retrieve_user=self.auth.retrieve_user,
                   user_id='id',
                   secret='SmartEye!08061999',
                   inject_user=True,
                   )

        self.server = self.sanic.create_server(
            host='0.0.0.0',
            port=8080,
            return_asyncio_server=True,
            debug=False
        )

        self.loop.create_task(self.server)

    def add_socket(self, socket):
        self.sockets.append(socket)

    def remove_socket(self, socket):
        try:
            self.sockets.remove(socket)
        except ValueError:
            pass  # already removed

    async def send_all_sockets(self, data):
        data = json.dumps(data)
        for socket in self.sockets:
            try:
                await socket.send(data)
            except ConnectionClosed:
                self.remove_socket(socket)

