# SharpEye
SharpEye Venture
Windows/mac install Linux environment
1. Download and install Oracle virtual machine
2. Download and install Ubuntu 18.04.1 LTS on an Oracle VM
3. Download and install Oracle Extension Package

On Linux:
1. install openCV (http://www.codebind.com/linux-tutorials/install-opencv-ubuntu-18-04-lts-python/)
2. Add these lines before step 2 in the above link:

    a. sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
    
    b. sudo apt update
    
    c. sudo apt install libjasper1 libjasper-dev
    
    d. sudo apt-get install libcanberra-gtk-module:i386

    e. in step 2 of link replace python3.5-dev with python 3.6-dev
