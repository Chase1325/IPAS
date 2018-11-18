import socket
import json


class Localization:
    def on_get(self, req, resp):
        """Return current location to client."""
        pos = None
        if socket.gethostname().lower() in ('cera', 'littlefoot'):
            # Ground vechicle, call on the RobotControl Server to move
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server = ('localhost', 6579)
            client.settimeout(None)
            client.connect(server)
            client.sendall('pos\n')
            while True:
                try:
                    data = self.conn.recv(1)
                    while data and data[-1] != ord('\n'):
                        data += self.conn.recv(1)
                    if not data:
                        break
                except socket.error:
                    break

                try:
                    text = data.decode().strip()
                except UnicodeDecodeError:
                    break
                
                if text.lower() == 'done':
                    if pos:
                        resp.status = falcon.HTTP_200
                    else:
                        resp.status = falcon.HTTP_500
                    break
                else:
                    try:
                        x, y = (int(i) for i in text.strip('()').split(','))
                    except:
                        resp.status = falcon.HTTP_500
                        break
                    else:
                        pos = {'x': x, 'y': y}
        else:
            # UAV
            pass

        resp.body = json.dumps(pos)
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        """Move to specified location."""
        def get_coord(req, coord):
            try:
                return int(req.get_param(coord))
            except (ValueError, falcon.HttpInvalidHeader):
                return None
        x = get_coord(req, 'x')
        y = get_coord(req, 'y')
        z = get_coord(req, 'z')
        if socket.gethostname().lower() in ('cera', 'littlefoot'):
            # Ground vechicle, call on the RobotControl Server to move
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server = ('localhost', 6579)
            client.settimeout(None)
            client.connect(server)
            client.sendall(f'goto:({x},{y})\n')
            try:
                data = self.conn.recv(1)
                while data and data[-1] != ord('\n'):
                    data += self.conn.recv(1)
                if not data:
                    raise socket.error
            except socket.error:
                resp.status = falcon.HTTP_500
                return

            try:
                text = data.decode().strip()
            except UnicodeDecodeError:
                resp.status = falcon.HTTP_500
                return
            
            if text.lower() == 'done':
                resp.status = falcon.HTTP_200
            else:
                resp.status = falcon.HTTP_500
        else:
            # UAV
            pass

