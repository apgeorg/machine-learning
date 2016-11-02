#!/bin/bash

echo "Downloading flower photos"
wget -c http://download.tensorflow.org/example_images/flower_photos.tgz

echo "Extracting flower photos"
tar xzf flower_photos.tgz
