#!/bin/bash
set -e

docker run --name autodrive_roboracer_api \
  --rm -it \
  --network=host \
  --ipc=host \
  -v $PWD:/home/autodrive_devkit:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
  --privileged \
  --gpus all \
  autodriveecosystem/autodrive_roboracer_api:2026-icra-practice

