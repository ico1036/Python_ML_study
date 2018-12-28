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

