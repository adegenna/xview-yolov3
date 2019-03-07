Installation
=====================

The purpose of this section is to provide detailed, step-by-step
instructions on how to install Anaconda for Python virtual
environments, the Pytorch framework, Nvidia GPU drivers, and the
YOLOv3 project repository.

Anaconda
---------------------

We first need to install Anaconda for Python virtual environments.

#. Download the Anaconda installer (shell script) from the Anaconda website::

     >>> wget https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh

   Note: this assumes you have a 64-bit Linux architecture. If you have something else, then visit https://www.anaconda.com/download/ and select your preferred version.

#. Launch the Anaconda installer::

     >>> bash Anaconda2-5.3.0-Linux-x86_64.sh

   Accept the user terms and accept the default filepath for installation, which should be ``/home/[user]/anaconda2/``.

#. Open your ``∼/.bashrc`` file in a file editor (e.g., ``emacs ∼/.bashrc``) and paste the following line to the end::

     >>> source /home/[user]/anaconda2/etc/profile.d/conda.sh

#. Save the ``∼/.bashrc`` file, exit, and reload it in your terminal with::

     >>> source ∼/.bashrc
   
#. Confirm conda was installed::

     >>> conda --version

   This should output the version of the Anaconda install, if successful

#. Create a custom Anaconda virtual environment for this project::

     >>> conda create -n [envname] python=3.6 anaconda

   In the above, replace [envname] with your desired environment name (do not include the brackets)

#. To verify that this was successful, run::

     >>> conda info --envs

   If successful, [envname] should appear as one of the choices.

PyTorch
---------------------

We will now install PyTorch, a Python deep-learning framework

#. Install PyTorch/Torchvision to your Anaconda environment::

     >>> conda install -n [envname] pytorch torchvision -c pytorch

#. To verify that this was successful, activate your conda environment::

     >>> conda activate [envname]

   Then, check the PyTorch version with::

     >>> python -c "import torch; print(torch.__version__)"

   Also check the Torchvision version with::

     >>> python -c "import torchvision; print(torchvision.__version__)"

If successful, both commands should output the installed versions.

GPU Support
---------------------

.. note:: 
   This section is only necessary if you have Nvidia GPU hardware, but have not yet installed drivers for it. Please be aware that installing Nvidia drivers can be tricky and should be handled with care. This section is a guide only; a thorough description of how to install GPU drivers is outside of the scope of this project.

.. note:: 
   These instructions may require sudo priveleges.

.. note:: 
   These instructions assume a Redhat OS. The equivalent process for another Linux OS (e.g., Ubuntu) is very similar.

#. Prepare your machine by installing necessary prerequisite packages::

     >>> yum -y update

     >>> yum -y groupinstall "Development Tools"

     >>> yum -y install kernel-devel epel-release

     >>> yum install dkms

#. Download desired Nvidia driver version from their archive at https://www.nvidia.com/object/unix.html (e.g., using wget from the terminal)

#. If your machine is currently using open-source drivers (e.g., noveau), you will need to change the configuration ``/etc/default/grub`` file. Open this file, find the line beginning with ``GRUB_CMDLINE_LINUX`` and add the following text to it::

     nouveau.modeset=0

4. Reboot your machine

5. Stop all Xorg servers::

     >>> systemctl isolate multi-user.target

6. Run the bash script installer::

     >>> bash NVIDIA-Linux-x86_64-*

7. Reboot your system

8. Confirm that the installation was successful by inspecting the output of this command::

     >>> nvidia-smi
   
   If successful, this should display all Nvidia GPUs currently installed in your machine

YOLOv3
---------------------

Note: For now, we are simply using a version of YOLOv3 freely available on Github. We plan to fork this and modify it as needed. For now, we only describe the installation directions for the community-available version of YOLOv3.

#. Activate your anaconda environment::

     >>> conda activate [envname]

#. Clone the YOLOv3 git repo::

     >>> git clone https://github.com/adegenna/yolov3

#. All of Python packages listed in the Requirements section of this documentation must be installed to your local conda environment. You may check whether the listed packages are installed with::

     >>> conda list | grep [package]

#. If one of the required packages is missing, then install it; for example, install opencv-python with::

     >>> conda install -n [envname] -c menpo opencv
