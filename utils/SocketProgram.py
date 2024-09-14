import socket
import yaml
import time
import os
import sys

class SocketConnection:
    def __init__(self, host, port=3000):
        ''' Socket connection with 2 param
            host: your ip address host pc
            port: initiate port no above 1024. Default is 3000
        '''
        self.host = host
        self.port = port
        self.instance = socket.socket()  # Instantiate
        self.instance.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Reuse address
        
    def clientConnect(self):
        ''' Parse connect client socket via host and port
        '''
        while True:
            try:
                self.instance.connect((self.host, self.port))
                print(f"Connected to {self.host} on port {self.port}")
                break
            except socket.error as e:
                print(f"Failed to connect to {self.host} on port {self.port}: {e}")
                time.sleep(2)
                # self.clientConnect()
            except Exception as e:
                print(f"Failed to connect: {e}")
                time.sleep(2)

    def clientSend(self, message):
        # message = input("send: ")  # take input
        return self.instance.send(message.encode())  # send message
    
    def clientRcv(self):
        return self.instance.recv(1024).decode()  # receive response

    def serverBind(self, num=2):
        ''' 
            bind host address and port together
            The bind() function takes tuple as argument
            default configure number client the server can listen simultaneously
            global conn 
        '''
        while True:
            try:
                self.instance.bind((self.host, self.port))
                # configure how many client the server can listen simultaneously
                self.instance.listen(num)
                self.conn, address = self.instance.accept()  # accept new connection
                print("Connection from: " + str(address))
                break
            except socket.error as e:
                print(f"Failed to connect to {self.host} on port {self.port}: {e}")
                time.sleep(2)
                # self.serverBind()
            except Exception as e:
               print(f"Failed to connect: {e}")
               time.sleep(2)
    
    def serverRcv(self):
        ''' Receive response from client ("utf-8") '''
        return self.conn.recv(1024).decode() 
            
    def serverSend(self, message):
        ''' send message to the client ("utf-8")[:1024] '''
        return self.conn.send(message.encode())  

    def send(self, connType, message):        
        # return self.conn.send(message.encode()) if connType == "server" else self.instance.send(message.encode())
        if connType == "server":
            self.conn.send(message.encode())  # send message with server role sendall
        else:
            self.instance.send(message.encode())   # send message with client role

    def close(self):
        self.instance.close()
        print("\nSocket closed!")

if __name__ == '__main__': 
    host = "192.168.1.6" # IP address from your host pc 
    port = 3000  # socket server port number

    def getAbsPath(path):
        # Get absolutely path
        currentDir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(currentDir, path))

    path = getAbsPath("../config.yml") if len(sys.argv) <= 1 else sys.argv[1]
    with open(getAbsPath(path), 'r') as f:
        items = yaml.load(f, Loader=yaml.FullLoader)

    client = SocketConnection(items['Host'], items['Port'])
    client.clientConnect()
    while True:
        try:
            mess = input("-> ") or "sorry"
            t0 = time.time()
            client.clientSend(mess)
            data = client.clientRcv()
            print('Received from server: ' + data)  # show in terminal
            print(f'Done. ({time.time() - t0:.5f}s)')
        except KeyboardInterrupt:
            client.close()
            break

    # # Test connect client with server role

    # server = SocketConnection(items['Host'], 3000)   # 192.168.1.8
    # print("Waiting connection!")
    # server.serverBind()
    # while True:
    #     try:
    #         data = server.serverRcv() 
    #         t0 = time.time()
    #         print("Reveived from client: " + str(data))
    #         mess = "ok"
    #         server.send("server", str(mess))
    #         print(f'Done. ({time.time() - t0:.5f}s)')
    #     except KeyboardInterrupt:
    #         server.close()
    #         break