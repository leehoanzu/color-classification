import socket
import time

def server_program():
    host = "192.168.0.100"
    port = 2999 # initiate port no above 1024
    server_socket = socket.socket()  # get instance
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Reuse address
    
    try:
        server_socket.bind((host, port))  # bind host address and port together
    except socket.error as e:
        print(f"Error binding to socket: {e}")
        return

    server_socket.listen(2)
    print(f"Server listening on {host}:{port}")

    try:
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
    except socket.error as e:
        print(f"Error accepting connection: {e}")
        return

   # Example data to send
    data = "1"

    while True:
        try:
            conn.send(data.encode('utf-8')) 
            print("\nHave been sending: " + str(data.encode('utf-8')))
                
            msg = conn.recv(1024).decode('utf-8')
            
            if not msg:
                # if data is not received, print and break
                print("No data received. Closing connection.")
                break
            print("from connected user: " + str(msg))

            time.sleep(1)
        except socket.error as e:
            print(f"Error during communication: {e}")
            break
        except KeyboardInterrupt:
            break

    conn.close()  # close the connection


if __name__ == '__main__':
    server_program()
