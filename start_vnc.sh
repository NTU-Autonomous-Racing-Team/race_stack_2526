#!/bin/bash

# Configuration
RESOLUTION="1280x720x24"
DISPLAY_NUM=":0"
VNC_PORT="5900"

echo "Starting Xvfb on display $DISPLAY_NUM..."
Xvfb $DISPLAY_NUM -screen 0 $RESOLUTION -ac +extension GLX +render -noreset &
sleep 2

echo "Starting fluxbox..."
DISPLAY=$DISPLAY_NUM fluxbox &
sleep 1

echo "Starting x11vnc on port $VNC_PORT..."
x11vnc -display $DISPLAY_NUM -forever -shared -bg -rfbport $VNC_PORT -nopw -xkb

echo "VNC Setup Complete. Connect to port $VNC_PORT."
