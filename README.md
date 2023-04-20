# PRJ
B.2 Installation
Installing the third party packages in a virtual environment will not affect the current environ-
ment. So the suggestion is to get a conda environment before installing any needed packages.
B.2.1 Installing Conda Environment
You can install Anaconda through https://www.anaconda.com/products/distribution
or install Miniconda through https://docs.conda.io/en/latest/miniconda.html.
The following steps is the guidance of how to manage the virtual environment in your com-
puter. To see more information, please refer to: https://conda.io/projects/conda/en/latest/user-
guide/tasks/manage-environments.html.
After installing the anaconda or miniconda, you can start to create your virtual environment
by typing in the terminal:
conda create --name myenv
Then, activate the environment using this command line:
conda activate myenv
If the environment is successfully activated, you should see:
(myenv) user@hostname: 
B.2.2 Installing the Third Party Packages
A requirements.txt is available for installing the needed packages. You could see the package
version and the package name in this file. To install all packages in one step, a file called
environment.yaml is provided. The file environment.yaml lists the necessary packages and
the corresponding versions for a specific software project to operate.
To install the requirements in environment.yaml, you can type the following command line:
conda env create -f environment.yaml
If anything failed to be installed, you can check the requirement version in the require-
ments.txt to install all these requirement manually by using pip install.
B.3 Getting Started
warning: the autotuning system might use some Ubuntu commands during training, please
try it on the Ubuntu system to make sure everything works well.
To check if everything is ready to run, please use under PRJ/src directory:
python3 test_cases.py
If all tests are passed, you can start by running a benchmark program for fun. The test-
ing command.txt contain all commands of each autotuning method for each benchmark pro-
gram on each row, copy you favourite command and paste it to the terminal window to run
the benchmark.
By follow this format, you are able to tuning your program automatically.
python3 main.py
--total-iters [number]
--autotuning-method [boca, bocaso or rio]
--flags-dir [dir] --file-cmd [" -o|-I exe_cmd dir_to_program"]
--baseline-cmd ["gcc -O3 -o|-I exe_cmd dir_to_program"]
--exe-cmd [exe_cmd]
--evaluate-times [number]
A example command:
python3 main.py --total-iters 60 --autotuning-method bocaso
--flags-dir "../flaglist/gcc7.5.txt"
--file-cmd " -o hello_world Test_C_Programs/hello_world.c"
--baseline-cmd "gcc -O3 -o hello_world Test_C_Programs/hello_world.c"
--exe-cmd ./hello_world
--evaluate-times 2
Please use [your command] if your input arguments contains whitespaces.
To see help message use:
python3 main.py --help
You may like to try the optimisation flags that your GCC compiler support, just put the flags
into a txt file and spilt them by whitespaces, and then input the directory using the format
above.