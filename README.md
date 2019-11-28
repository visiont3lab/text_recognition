# Text Recognition in Natural Scene Images


## Introduction
Deep Learning Text recognition based on CRAFT (Character-Region Awareness For Text detection)

## Requirements

* [Install docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)
* [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)

## Setup

### Cloning repo
```
cd $HOME && \
git clone https://github.com/visiont3lab/text_recognition.git && \
echo "export TEXT_RECOGNITION=$HOME/text_recognition" >> $HOME/.bashrc && source $HOME/.bashrc
```

### Download models
```
cd $TEXT_RECOGNITION/code && mkdir -p model && cd model && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies  \
    /tmp/cookies.txt --keep-session-cookies --no-check-certificate  \
    'https://docs.google.com/uc?export=download&id=1gjePriCBUC8TDZbNEicojxfSA5oTGqPK' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gjePriCBUC8TDZbNEicojxfSA5oTGqPK" \
    -O craft_mlt_25k.pth && rm -rf /tmp/cookies.txt && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies \
    /tmp/cookies.txt --keep-session-cookies --no-check-certificate  \
    'https://docs.google.com/uc?export=download&id=11TpvsQuJtOeacKrXl4vgEH4VHg4PVRFv' \
    -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=11TpvsQuJtOeacKrXl4vgEH4VHg4PVRFv" \
    -O TPS-ResNet-BiLSTM-Attn.pth && rm -rf /tmp/cookies.txt
```

## Run

```
xhost +local:docker && \
    docker run --runtime=nvidia  --rm  \
        -it --name deep_learning_face_recognition  \
        --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --device=/dev/video0  \
        -v $TEXT_RECOGNITION:/root/home/ws \
        visiont3lab/deep-learning:all \
        /bin/bash -c "cd /root/home/ws/code/ && python3 demo3.py"
```

## References
[Reference Repository](https://github.com/clovaai/CRAFT-pytorch)
