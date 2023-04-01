from os import path
from sanic.response import json


def setup_routes(web, app):
    sanic = web.sanic

    @sanic.route("/", methods=['GET'])
    async def handler(req):
        return json(status=200, body={'result': 'OK'})
