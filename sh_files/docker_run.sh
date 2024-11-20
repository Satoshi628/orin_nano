sudo docker run --runtime nvidia -it --rm --network=host \
    -v /home/rb53262/Downloads:/home/rb53262/Downloads \
    dustynv/onnxruntime:r35.4.1

pip install pyrealsense2