# Welcome to Color Classification Repository

This repository provides software that utilizes an AI model on the Raspberry Pi 5, connected to a PLC via Ethernet, for color classification. You will find in this repo the following stuff:

<a><img align="right" width="200" height="200" src="https://github.com/leehoanzu/color-classification/blob/main/screen-shots/panel.png"></a>

1. **Training the Model**: Offering a detailed guide on [trainning model](https://github.com/leehoanzu/color-classification/blob/main/docs/train.md) effectively to classify colors, with complete documentation available.
2. **Deploying the Model**: Efficiently [deploy the model](https://github.com/leehoanzu/color-classification/blob/main/docs/deploy.md) using ONNX Runtime by following the provided guidelines for conversion, installation, inference, and optimization.
3.  **Setting up communication**: Ensuring efficient communication between systems, we include a guide on setting up a [socket connection](https://github.com/leehoanzu/color-classification/blob/main/docs/socket.md) for data transfer between devices.
4. **Packaging into Docker**: Guidance on packaging the application into a [Docker container](https://github.com/leehoanzu/color-classification/blob/main/docs/packages.md) for ease of deployment and scalability.

## Quickstart

For detailed Docker setup instructions, refer to [packaged.md](https://github.com/leehoanzu/color-classification/blob/main/docs/packages.md).

```bash
$ git clone https://github.com/leehoanzu/color-classification.git
$ ./run.sh leehoanzu/raspberrypi-utils:latest
```

Or you can manually [run](https://github.com/leehoanzu/color-classification/blob/main/docs/deploy.md) a main script:

1. **Clone the Repository**

    First, clone the repository from GitHub:

    ```bash
    $ git clone https://github.com/leehoanzu/color-classification.git
    $ cd color-classification
    ```

2. **Create a Virtual Environment**

    It's recommended to create a virtual environment to isolate your project dependencies:

    ```bash
    # Create virtual environment
    $ python3 -m venv onnx_venv

    # Activate the virtual environment
    $ source onnx_venv/bin/activate
    ```

> [!NOTE]  
> <sup>- `onnx_venv` is path to new virtual environment.</sup>

3. **Install Dependencies**

    After activating the virtual environment, install the required dependencies:

    ```bash
    $ pip install -r requirements.txt
    ```

    This will install all the packages needed to run the project, as specified in the [requirements.txt](https://github.com/leehoanzu/color-classification/blob/main/requirements.txt) file.

4. **Run the Main Script**

    Once everything is set up, you can run the main script:

    ```bash
    $ python3 ./utils/main.py
    ```

    This command will execute the main Python script located in the `utils` directory.

## Contributing

* Contributions are welcome! If you want to contribute to the project, feel free to submit a pull request. Be sure to follow the existing style and include detailed commit messages.

## Contact

* Contact with me via email: lehoangvu260602@gmail.com

## Copyright

* Copyright &#169; 2024 Lê Hoàng Vũ