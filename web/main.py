from sanic import Sanic
from sanic_cors import CORS

from web.plugins import jwtPlugin
from web.plugins.middlewares import setup_middlewares
from web.routes import setup_routes
from web.settings import Settings


class WebServer:
    def __init__(self):
        sanic = Sanic("Auto-Charger")
        sanic.config.update(Settings)

        jwtPlugin.setup_jwt(sanic)
        CORS(sanic)

        setup_routes(sanic)
        setup_middlewares(sanic)

        sanic.run(
            host=sanic.config.HOST,
            port=sanic.config.PORT,
            debug=sanic.config.DEBUG,
            auto_reload=sanic.config.DEBUG,
        )
