import os
from datetime import datetime
from json import dumps
from os import path
import sys
from sanic.response import file, json
from sanic.exceptions import NotFound, ServerError
from websockets.exceptions import ConnectionClosed


def load_sys_routes(web, app):
    root = app.root
    print(root)
    sanic = web.sanic

    public = os.path.join(root, 'public')
    assets = os.path.join(public, 'static')
    sanic.static('/static', assets)

    @sanic.route("/", methods=['GET'])
    async def handler(req):
        tmpl_file_path = path.join(public, 'index.html')
        return await file(tmpl_file_path)

    @sanic.route('/favicon.ico')
    async def index(request):
        tmpl_file_path = path.join(public, 'favicon.ico')
        return await file(tmpl_file_path)

    @sanic.exception(NotFound)
    async def json_404s(request, exception):
        if request.method == 'GET':
            return await file(path.join(public, 'index.html'))
        return {'error': exception}

    @sanic.exception(ServerError)
    async def json_500s(request, exception):
        return json({'error': exception}, status=500)

    @sanic.websocket('/websocket')
    async def feed(request, ws):
        web.add_socket(ws)

        # if os.getenv('TELEMETRY') == 'fake':
        #     await web.fake.telemetry(web)
        while True:
            try:
                message = await ws.recv()
            except ConnectionClosed:
                web.remove_socket(ws)
                break
            else:
                await web.send_all_sockets(dumps({
                    'action': 'pong',
                    'timestamp': datetime.now().isoformat(),
                }))
