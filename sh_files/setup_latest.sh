# get nvidia-jetpack
sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/common r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'
sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/t234 r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'

sudo apt -y update
sudo apt -y install nvidia-jetpack

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# get realsense
git clone https://github.com/jetsonhacksnano/installLibrealsense
cd installLibrealsense
sh ./installLibrealsense.sh

realsense-viewer

cd ..

# mosquitto install
sudo apt-get -y install mosquitto mosquitto-clients



############ pyrealsense2 ############
# apt-get install libssl-dev xorg-dev libglu1-mesa-dev
sudo apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev 
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get -y install python3 python3-dev libssl-dev libxinerama-dev libxcursor-dev libcanberra-gtk-module libcanberra-gtk3-module
git clone https://github.com/IntelRealSense/librealsense/ -b v2.50.0

#CmakeLists.txt edit
#https://github.com/IntelRealSense/librealsense/issues/7829

cd librealsense 
mkdir build && cd build

CUDACXX=/usr/local/cuda-11.4/bin/nvcc cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true -DFORCE_RSUSB_BACKEND=ON

sudo make uninstall && make clean

make -j6  

sudo make install

# in ~/.bashrc
echo 'PYTHONPATH="/usr/local/lib:/usr/local/lib/python3.8/pyrealsense2:$PYTHONPATH"' >> ~/.bashrc
echo 'export PYTHONPATH' >> ~/.bashrc


sudo mkdir /usr/local/lib/python3.8/pyrealsense2

sudo ln -s wrappers/python/pybackend2.cpython-38-aarch64-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/

sudo ln -s wrappers/python/pyrealsense2.cpython-38-aarch64-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/

source ~/.bashrc
sudo apt-get install python3-pip
python3 -m pip install numpy opencv-python pyrealsense2 paho-mqtt onnxruntime==1.18.0

cd ..
cd ..
cd code
sudo chmod 744 main.py
# set up network
# settings->Network->PCI Ethernet->IPv4->Manual
# 192.168.1.153 255.255.255.0

# start up command
echo -e "[Unit]\nDescription=start mos\nAfter=network.target\n\n[Service]\nUser=rb53262\nExecStart=systemctl restart mosquitto\nType=simple\n\n[Install]\nWantedBy=multi-user.target\n" | sudo tee /etc/systemd/system/mosquitto.service
echo -e "[Unit]\nDescription=start python\nAfter=network.target\n\n[Service]\nUser=rb53262\nExecStart=/usr/bin/python3 /home/rb53262/Desktop/orin_nano/code/main.py\nType=simple\n\n[Install]\nWantedBy=multi-user.target\n" | sudo tee /etc/systemd/system/senser.service

sudo systemctl enable mosquitto.service
sudo systemctl enable senser.service

