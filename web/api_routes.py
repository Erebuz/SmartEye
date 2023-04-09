import os
from os import path
from sanic.response import file, json
from sanic.exceptions import NotFound, ServerError, InvalidUsage
from sanic_jwt import inject_user, protected


def load_api_routes(web, app):
    sanic = web.sanic

    @sanic.route('/api/users', methods=['PUT'])
    @protected()
    @inject_user()
    async def handler(request, user):
        result = await web.auth.update_user(user['id'], request.json)

        return json('OK', status=200)

    @sanic.route("/api/video/fps", methods=['GET', 'OPTIONS'])
    async def handler(req):
        return json({
            "current": app.RtspMergeServer.current_fps,
            "max": app.RtspMergeServer.fps_max,
            "target": app.RtspMergeServer.target_fps
        }, status=200)

    @sanic.route("/api/video/fps", methods=["PUT"])
    async def handler(req):
        body = req.json
        val = body.get('target', None)
        if val is not None:
            app.RtspMergeServer.set_target_fps(val)
            return json("Target FPS updated", status=200)
        else:
            raise InvalidUsage

    @sanic.route("/api/nn", methods=['GET', 'OPTIONS'])
    async def handler(req):
        result = {
            "show_osd": app.RtspMergeServer.show_osd,
            "classes": app.RtspMergeServer.supported_class
        }
        return json(result, status=200)

    @sanic.route("/api/nn", methods=["PUT"])
    async def handler(req):
        body = req.json
        enable = body.get('enable', None)
        classes = body.get('classes', None)

        if enable is not None:
            app.RtspMergeServer.show_osd = enable

        if classes is not None:
            app.RtspMergeServer.supported_class = classes

        return json("NN status updated", status=200)

    @sanic.route("/api/nn/skip", methods=['GET', 'OPTIONS'])
    async def handler(req):
        return json(app.RtspMergeServer.frame_skip, status=200)

    @sanic.route("/api/nn/skip", methods=["PUT"])
    async def handler(req):
        body = req.json
        val = body.get('skip', None)
        if val is not None:
            app.RtspMergeServer.frame_skip = int(val)
            return json("Frame skip updated", status=200)
        else:
            raise InvalidUsage

    @sanic.route("/api/nn/log", methods=['GET', 'OPTIONS'])
    async def handler(req):
        return json(app.RtspMergeServer.activity_log, status=200)
