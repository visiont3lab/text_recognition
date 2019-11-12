# Text Recognition

## Introduction
Deep Learning Text recognition based on CRAFT (Character-Region Awareness For Text detection)

## Requirements

* [Install docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)
* [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)
* [Download Models](https://drive.google.com/drive/folders/1MUZJfJaErK5UHwQAFaf5dHdeQfaJL09-?usp=sharing) Put the two folders (models+model inside text_recognition/code of this repo).

## Setup

```
cd $HOME
git clone https://github.com/visiont3lab/text_recognition.git
echo "export TEXT_RECOGNITION=$HOME/text_recognition" >> $HOME/.bashrc && source $HOME/.bashrc
```

## Run

```
xhost +local:docker && \
    docker run --runtime=nvidia  --rm  \
        -it --name deep_learning_face_recognition  \
        --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device=/dev/video0  \
        -v $TEXT_RECOGNITION:/root/home/ws \
        visiont3lab/deep-learning:text_recognition \
        /bin/bash -c "cd /root/home/ws/code/ && python3 demo3.py"
```

## References
[Reference Repository](https://github.com/clovaai/CRAFT-pytorch)
