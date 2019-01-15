## Local installation  
### 1. Anaconda Installation  
  
```bash
$ wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh  
$ chmod +x Anaconda2-4.3.1-Linux-x86_64.sh  
$ ./Anaconda2-4.3.1-Linux-x86_64.sh  
$ export PATH=/home/[user_name]/anaconda/bin:$PATH
```  
 - test    
```bash
$ conda  
```  
: man page open  
```bash
$ which pip  
```  
: ~/anaconda/bin/pip  

### 2. Tensorflow installation under conda enviroment  

 - create new conda enviroment named tensorflow  
```bash
$ conda create -n tensorflow python=2.7
```  

 - activate the conda enviroment  
```bash
$ source activate tensorflow
```  

 - install tensorflow ...[All releases](https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package)  
``` bash
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp27-none-linux_x86_64.whl  
$ pip install --upgrade $TF_BINARY_URL
```  

 - deactivate the conda enviroment  
```bash
$ source deactivate tensorflow
```  

 - test  
```bash
$ source activate tensorflow
$ python
```  
```python
import tensorflow as tf
```  
: No error -> success  

### 3. Syncronize jupyter notebook with tensorflow-conda enviroment  
```bash
$ source activate tensorflow  
$ python -m ipykernel install --user --name tensorflow --display-name "Python (tensorflow)"
$ jupyter notebook
```  
: When you click the [new] you can find the Python(tensorflow)
  
  
-Reference: [Ref](http://shilan.tistory.com/entry/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0-CentOS-71Python-27Anaconda)  

---
---
  
## Docker  

This is fast installation of tensorflow using docker  
The basic understanding of docker must be preceded  
Read the [reference](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)  
### 1. Download a docker  

```bash
$ curl -fsSL https://get.docker.com/ | sudo sh
```  
 - Grant autority to users
```bash
$ sudo usermod -aG docker #USER
```  
 - Start docker and check
```bash
$ sudo systemctl start docker
$ docker version
```  
output: Client: ~~~ Server: ~~~  

### 2. Download a tensorflow with jupyternotebook image  

```bash
$ docker run -u $(id -u):$(id -g) -it -p 8888:8888 tensorflow/tensorflow:nightly-py3-jupyter
```  
  

### 3. Control container  
 - Restart a container
```bash
$ docker start -a containerID
```  
 - You can find the name and ID of the ended container as follwing
```bash
$ docker ps -a
```  
- You cna find the name of the images as follwing
```bash
$ docker images
 _REPOSITORY is a name of image_
```  

## Tensorflow GPU  
### 1. Nvidia drvier installation  
__Beware the kernel panic!! First, try it using test-cpu__  
 - [Download](https://www.nvidia.com/Download/index.aspx?lang=en-us ) suitable driver file  
 __!!!Next step can make Kernel panic! Possible solutions are discribed in (a) and (b)!!!__  
---
 - Rest steps for installation [Link](http://linux.systemv.pe.kr/nvidia-driver-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0/
)  
 a) __Reboot and press [f2] to enter BIOS and disable the secure booting__( This works well for me to escape from kernel panic )  
#### b) __Change into run level3__( [artl]+[F3] ) __and re-install the Nvidia driver__ ( general solutions )  

### 2. Nvidia-docker installation  
 
 - You need to have "docker-ce-3:18.09.0.el7" version  [DownloadLink](https://docs.docker.com/install/linux/docker-ce/centos/
)
 - Install the nvida-docker-2 [DownloadLink](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
 - When the gpu information appear, suecess!

### 3. Tensorflow-gpu-image installasion
 - You don not need to install the cuda using the Docker
```bash
docker run --runtime=nvidia -it -p 8888:8888 tensorflow/tensorflow:latest-gpu
```  
 - go to the url(it may be http://127.0.0.1:8888) and type the token which appear on yout terminal
 - Test the cpu,gpu in jupyter notebook
```python
import tensorflow as tf
from tensorflow.python.client import device_lib
evice_lib.list_local_devices()
```  
: cpu and gpu information appear -> sucess 

