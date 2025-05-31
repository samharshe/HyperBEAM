#!/bin/bash

JPEG_FILE="3.jpeg"

URL="http://127.0.0.1:3000/infer"

curl -X POST "$URL" \
     -H "Content-Type: image/jpeg" \
     --data-binary @"$JPEG_FILE"