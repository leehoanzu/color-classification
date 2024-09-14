# Socket Connection

Socket programming is a way of connecting two nodes on a network to communicate with each other. One socket (node) listens on a particular port at an IP, while the other socket reaches out to the other to form a connection. The server forms the listener socket while the client reaches out to the server.

## State Diagram for Server and Client Model

![state diagram](https://github.com/leehoanzu/angle-detection/blob/main/screen-shots/diagram_socket.png)

## Stages for Server

### Create Socket

```python
import socket

# Create socket instance
server_socket = socket.socket() 
```

### Bind

You can set the IP and port manually or load them from [`config.yml`](https://github.com/leehoanzu/color-classification/blob/main/config.yml):

```python
import sys
import yaml

# Load IP and port from config.yml or command-line arguments
path = getAbsPath("../config.yml") if len(sys.argv) <= 1 else sys.argv[1]

with open(getAbsPath(path), 'r') as f:
        items = yaml.load(f, Loader=yaml.FullLoader)

# host = "192.168.1.5"
# port = 5000  # initiate port no above 1024
#server_socket.bind(host, port)

server_socket.bind(items['Host'], items['Port'])
```

### Listen

```python
# number client is accepted
server_socket.listen(2)
```

### Accept

```python
# accept new connection
# conn: instace of connection
# address: address connection

conn, address = server_socket.accept() 
```

## Stages for Client

### Create Socket

```python
import socket

# Create socket instance
client_socket = socket.socket()
```

### Connect

You can use  [`config.yml`](https://github.com/leehoanzu/color-classification/blob/main/config.yml) or command-line arguments to configure the IP and port:

```python
import sys
import yaml

# Load IP and port from config.yml or from command-line arguments
path = getAbsPath("../config.yml") if len(sys.argv) <= 1 else sys.argv[1]
with open(getAbsPath(path), 'r') as f:
        items = yaml.load(f, Loader=yaml.FullLoader)

client = SocketConnection(items['Host'], items['Port'])

# host = "192.168.1.5"
# port = 5000  
# client_socket.connect((host, port))
```

> [!NOTE]  
> <sup>- The IP address and port can be configured via the [`config.yml`](https://github.com/leehoanzu/color-classification/blob/main/config.yml) file, and passed as an argument at runtime if needed..</sup>

## Run the Script

We have two methods to modify and provide the values at runtime:
    1. Modify the IP and port directly in the script.
    2. Edit the config YAML file and pass it via the terminal.

```bash
$ python3 SocketProgram.py  # Uses default "../config.yml"
```
Or
```bash
$ $ python3 SocketProgram.py ./path_to_config/config.yml  # Pass a custom config file path
```

## Reference

![`Result`](https://github.com/leehoanzu/angle-detection/blob/main/screen-shots/socket_connection.png)
