# CNN inference on GPU using tensorflow-gpu
The aim of this project is to perform inference of a CNN on GPU using Tensorflow.

## Setting up the environment

### Install Cuda

Each version of Tensorflow requires a specific version of Cuda. The following table allows you to find the Cuda version corresponding to the chosen Tensorflow version: [Compatibility matrix](https://www.tensorflow.org/install/source#gpu).

In this example, we will install and use Tensorflow 2.0.0 and Cuda 10.0.

**First**, start a terminal and remove any NVIDIA traces you may have on your machine.

```bash
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
```

**Then**, setup the correct CUDA PPA (Personal Package Archives) on your system.

```bash
sudo apt update
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

**Thirdly**, install the CUDA and cuDNN packages.

```bash
sudo apt update
sudo apt install cuda-10-0
sudo apt install libcudnn7
```

**Finally**, you need to specify PATH to CUDA in ‘.profile’ file. Open the file by running:

```bash
sudo vi ~/.profile
```

**And** add the following lines at the end of the file:

```bash
# set PATH for cuda 10.0 installation
if [ -d "/usr/local/cuda-10.0/bin/" ]; then
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

**Restart** and check the versions for the installation.

#### CUDA

```bash
nvcc  --version
```

>nvcc: NVIDIA (R) Cuda compiler driver
>Copyright (c) 2005-2019 NVIDIA Corporation
>Built on Wed_Apr_24_19:10:27_PDT_2019
>Cuda compilation tools, release 10.0, V10.0.130

#### NVIDIA Driver

```bash
nvidia-smi
```

If your shell returns an error saying that the command is not found, this means that the Nvidia graphics driver is not installed. You must therefore install it. Type the following command in a shell:

```bash
ubuntu-drivers devices
```

>WARNING:root:_pkg_get_support nvidia-driver-390: package has invalid Support Legacyheader, cannot determine support level
>== /sys/devices/pci0000:00/0000:00:03.1/0000:07:00.0 ==
>modalias : pci:v000010DEd00001380sv00001043sd000084BBbc03sc00i00
>vendor   : NVIDIA Corporation
>model    : GM107 [GeForce GTX 750 Ti]
>driver   : nvidia-driver-418-server - distro non-free
>driver   : nvidia-driver-390 - third-party non-free
>driver   : nvidia-driver-460-server - distro non-free
>driver   : nvidia-driver-450-server - distro non-free
>driver   : nvidia-driver-465 - third-party non-free
>driver   : nvidia-driver-410 - third-party non-free
>driver   : nvidia-driver-460 - third-party non-free **recommended**
>driver   : nvidia-340 - distro non-free
>driver   : xserver-xorg-video-nouveau - distro free builtin

Then you can use the following command to install the recommended driver:

```bash
sudo ubuntu-drivers autoinstall
```

If it fails, you can install the driver manually by doing:

```bash
sudo apt install nvidia-xxx
```

> Just replace 'xxx' by your specific Nvidia driver version, in my case it is nvidia-460

Once the installation is concluded, **reboot** your system.

#### libcuDNN

You can verify that cuDNN is properly installed with the following command:

```bash
/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```

>libcudnn.so.7 -> libcudnn.so.**7.6.5**

### Python setting up

Finally, you just have to use the Python version you want, according to the needed Tensorflow version.
The following commands allow you tu set up a virtual environnement which contains all the needed packages to run the CNN example.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
