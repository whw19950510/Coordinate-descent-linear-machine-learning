#!/bin/bash

# The ip of the remote server
REMOTE_IP="aorta@0.tcp.ngrok.io"
REMOTE_PORT="11980"
# Remote directory on the server that synchronization performs on
REMOTE_DIR="~/Documents/CS839/doubleworking"
# The absolute path of the local file that synchronization performs on
LOCAL_DIR="./"
LOCAL_FILE="techniques.cu"
SYNC_INTERVAL=5

while true
do
  scp -P$REMOTE_PORT $LOCAL_DIR/$LOCAL_FILE $REMOTE_IP:$REMOTE_DIR 
  sleep $SYNC_INTERVAL
done
