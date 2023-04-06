import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='Autotuning_System')
    parser.add_argument('--file-cmd', default = ' -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time', type=str,
                        help='Command for determining what file to execute (default is 2mm program in polybench)')
    parser.add_argument('--baseline-cmd', default = 'gcc -O2 -funswitch-loops -ftree-vectorize -fpredictive-commoning -fipa-cp-clone -finline-functions -fgcse-after-reload -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time', type=str,
                        help='Command for executing the baseline optimisation flags')
    parser.add_argument('--remove-cmd', default = 'rm -rf *.o *.I *.s a.out', type=str,
                        help='Command for removing the files that generated from the compilation')
    parser.add_argument('--time-cmd', default = './2mm_time', type=str,
                        help='Command for executing the optimised code, so that the time can be measured')
    parser.add_argument('--evaluate-times', default = 6, type=int,
                        help='It is about how many times for evaluating a flag sequence')
    parser.add_argument('--total-iters', default = 60, type=int,
                        help='the iteration times for the algorithm')
    parser.add_argument('--flags-dir', default = "../flaglist/gcc7.5flags.txt", type=str,
                        help='the txt file contains all optimisation flags')
    parser.add_argument('--autotuning-method', default='boca', type=str,
                        help='the method for tuning optimisation flags, options: boca, rio, bocaso (the default is boca)')
    args = parser.parse_args()

    return args
