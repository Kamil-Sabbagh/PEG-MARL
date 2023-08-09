# personalized-expert-guided-MARL

# Setup Tutorial

This tutorial will guide you through setting up the StarCraft Multi-Agent Challenge (SMAC) environment.

## Steps

1. **Install the Required Packages**

    First, we need to install the necessary requirements listed in `requirements.txt`. Run the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Set Up the Conda Environment for SMAC**

    If you do not have `conda` installed, follow these steps to install it:

    - Download [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh) or [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh).

    - Open your terminal and run one of the following commands:

        - For Miniconda:

            ```bash
            bash Miniconda3-latest-Linux-x86_64.sh
            ```

        - For Anaconda:

            ```bash
            bash Anaconda-latest-Linux-x86_64.sh
            ```

    - Follow the prompts on the installer screens.

    Now that we have `conda` installed, we can set up the SMAC environment:

    - Install the SMAC package from its git repository:

        ```bash
        pip install git+https://github.com/oxwhirl/smac.git
        ```

    - Create and activate a new conda environment named "SMAC" with Python version 3.10:

        ```bash
        conda create -n SMAC python=3.10
        conda activate SMAC
        ```

    - Install the necessary conda packages:

        ```bash
        conda install gxx_linux-64 gcc_linux-64 swig
        ```

    - Install the SMAC package:

        ```bash
        pip install smac
        ```

You should now have your SMAC environment set up.

