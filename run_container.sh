#!/bin/bash

DOCKER_IMAGE="thomas-blog-devel"
CONTAINER_NAME="thomas-blog-devel"
JEKYLL_PORT=4033

docker run -it \
    --rm \
    -p $JEKYLL_PORT:$JEKYLL_PORT \
    -e JEKYLL_PORT=$JEKYLL_PORT \
    --name $CONTAINER_NAME \
    --user $(id -u):$(id -g) \
    -v "$PWD":/srv/jekyll \
    $DOCKER_IMAGE /bin/bash
