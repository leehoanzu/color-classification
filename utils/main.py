import time

import cv2
import yaml
import os
import sys

from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from OnnxRuntime import OnnxUtils
from SocketProgram import SocketConnection

def threadPredict(img, connType, connector):
    # Start inferencing
    t0 = time.time()

    # Start inferencing
    imageY = ortSession.preProcessImage(img)
    outputY = ortSession.infer(imageY)
    
    # Transmit data
    connector.send(connType, f"{outputY}")
    ortSession.display(outputY)

    print(f'Done predict. ({time.time() - t0:.3f}s)')

def threadCap(connType, connector):
    global cap
    
    t0 = time.time()

    try:
        access, img = cap.read()

        if access:
            # Give the thread inference
            Thread(target=threadPredict, args=(img, connType, connector)).start()          
            cap.release()
            getCameraId()
        else:
            # print("Failed to capture image.\n")
            cap.release()
            time.sleep(1)
            getCameraId()  
            
            threadCap(connType, connector)
    except Exception:
        cap.release()
        time.sleep(1)
        # getCameraId()
        connector.send(connType, f"{1}")


def getAbsPath(path):
    ''' Get absolutely path '''
    currentDir = os.path.dirname(os.path.abspath(__file__))

    return os.path.abspath(os.path.join(currentDir, path))

def configYmlFile():
    '''
        - This is 2 method read file yaml config from local disk
        - Number 1 is loaded from local disk via function
        - Number 2 is given argrument via terminal console        
    '''
    path = getAbsPath("../config.yml") if len(sys.argv) <= 1 else sys.argv[1]
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def getCameraId():
    global cap

    while True:
        try:
            # Camera ID is 0
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 

            if cap is not None:
                cap.set(cv2.CAP_PROP_FPS, 36)
                break
            time.sleep(2)
        except cv2.error as e:
            # print(f"OpenCV Error: {e}")
            cap.release()
            time.sleep(1)
        except RuntimeError as e:
            # print(f"Runtime errror: {e}")
            cap.release()
            time.sleep(1)
        except Exception as e:
            # print(e)
            cap.release()
            time.sleep(1)


def clientProgram():
    # Make connection
    client = SocketConnection(items['Host'], int(items['Port']))
    client.clientConnect()

    while True:
        try:   
            # Listening signal from server         
            data = client.clientRcv() 
            if data.strip().lower() == items['Receive']:                
                # Wait for the task to complete
                executor.submit(threadCap, "client", client)
                time.sleep(1.5)
            else:
                # Close connection
                client.close() 
                time.sleep(1)
                return clientProgram()
        except Exception as e:
            client.close()
            return clientProgram()
        except KeyboardInterrupt:
            # Close connection
            client.close() 
            break
    

def serverProgram():
    # Make connection
    server = SocketConnection(items['Host'], int(items['Port']))

    # Bind host address and port together
    server.serverBind()

    while True:
        try:                
            data = server.serverRcv()
            if data.strip().lower() == items['Receive']:
                # Wait for the task to complete
                executor.submit(threadCap, "server", server)
                time.sleep(1.5)
            else:
                server.close()
                return serverProgram()
        except Exception as e:
            server.close()
            return serverProgram()
        except KeyboardInterrupt:
            # Close connection
            server.close()  
            break

def main():   
    getCameraId()
    print("\nStarting demo now! Press CTRL+C to exit\n")
    if items['Name'] == "server":
        serverProgram()    
    else: 
        clientProgram()
    # print("\nEnd.")


if __name__ == '__main__': 
    items = configYmlFile()
    # Print all the item and value in yaml file
    # for key, values in items.items():
    #     print(key + " : " + str(values))

    # Initialize the thread pool excutor
    executor = ThreadPoolExecutor(max_workers=3)

    # define path label and model
    label = getAbsPath("../label/color-label.txt")

    model = getAbsPath("../model/color-best-edc.onnx") 

    # Create intance of onnx-runtime
    ortSession = OnnxUtils(model, label)

    main()