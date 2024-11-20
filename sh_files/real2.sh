
apt update
# apt-get install libssl-dev xorg-dev libglu1-mesa-dev
sudo apt-get install -y git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev 
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
git clone https://github.com/IntelRealSense/librealsense/ -b v2.50.0

#CmakeLists.txt edit
#https://github.com/IntelRealSense/librealsense/issues/7829

cd librealsense 
mkdir build && cd build

cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j4
sudo make install

pip install opencv-python
pip install pyrealsense2


CUDACXX=/usr/local/cuda-11.4/bin/nvcc cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true -DFORCE_RSUSB_BACKEND=ON

sudo make uninstall && make clean

make -j6  

sudo make install

# in ~/.bashrc
PYTHONPATH="/usr/local/lib:/usr/local/lib/python3.8/pyrealsense2:$PYTHONPATH"
export PYTHONPATH


sudo mkdir /usr/local/lib/python3.8/pyrealsense2

sudo ln -s wrappers/python/pybackend2.cpython-38-aarch64-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/

sudo ln -s wrappers/python/pyrealsense2.cpython-38-aarch64-linux-gnu.so /usr/local/lib/python3.8/pyrealsense2/

source ~/.bashrc
