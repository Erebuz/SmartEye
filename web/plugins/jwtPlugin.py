from hashlib import sha256

from sanic_jwt import initialize, exceptions

from web.classes.User import User

users = [User(1, "admin", "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918", ['admin']),
         User(2, "user", "04f8996da763b7a969b1028ee3007569eaf3a635486ddab211d512c85b9df8fb", ["user"])]

username_table = {user.username: user for user in users}
userid_table = {user.user_id: user for user in users}
refresh_token_table = {}


async def authenticate(request, *args, **kwargs):
    username = request.json.get("username", None)
    password = sha256(bytes(request.json.get("password", None), encoding="utf-8")).hexdigest()

    user = username_table.get(username, None)

    if not username or not password:
        raise exceptions.AuthenticationFailed("Missing username or password.")

    if (user is None) or (password != user.password):
        raise exceptions.AuthenticationFailed("Authorization failed")

    return user


async def retrieve_user(request, payload, *args, **kwargs):
    if payload:
        user_id = payload.get('user_id', None)

        if user_id is None:
            return None

        user = userid_table.get(user_id)
        return user
    else:
        return None


async def get_user_roles(user, *args, **kwargs):
    return user.roles


def store_refresh_token(user_id, refresh_token, *args, **kwargs):
    key = 'refresh_token_{}'.format(user_id)
    refresh_token_table.update({key: refresh_token})


def retrieve_refresh_token(request, user_id, *args, **kwargs):
    key = f'refresh_token_{user_id}'
    return refresh_token_table.get(key)


def setup_jwt(app):
    initialize(app,
               authenticate=authenticate,
               retrieve_user=retrieve_user,
               add_scopes_to_payload=get_user_roles,
               secret="serenity",
               refresh_token_enabled=True,
               store_refresh_token=store_refresh_token,
               retrieve_refresh_token=retrieve_refresh_token
               )
