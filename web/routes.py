from sanic.response import json
from sanic_jwt.decorators import protected
from sanic_jwt.decorators import scoped


def setup_routes(app):
    @app.route("/", methods=["GET"])
    async def general(request):
        return json(status=200, body={"name": "general"})

    @app.route("/admin")
    @protected()
    @scoped("admin")
    async def admin(request, *args, **kwargs):
        return json(status=200, body={"name": "admin"})

    @app.route("/user")
    @protected()
    @scoped(["user", "admin"], False)
    async def user(request, *args, **kwargs):
        return json(status=200, body={"name": "user"})
