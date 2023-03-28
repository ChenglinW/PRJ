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
    parser.add_argument('--flags', default = ['-fno-peephole2', '-ffast-math', '-fno-schedule-insns2', '-fno-caller-saves', '-funroll-all-loops',
           '-fno-inline-small-functions', '-finline-functions', '-fno-math-errno', '-fno-tree-pre', '-ftracer',
           '-fno-reorder-functions', '-fno-dce', '-fipa-cp-clone', '-fno-move-loop-invariants', '-fno-regmove',
           '-funsafe-math-optimizations', '-fno-tree-loop-optimize', '-fno-merge-constants', '-fno-omit-frame-pointer',
           '-fno-align-labels', '-fno-tree-ter', '-fno-tree-dse', '-fwrapv', '-fgcse-after-reload', '-fno-align-jumps',
           '-fno-asynchronous-unwind-tables', '-fno-cse-follow-jumps', '-fno-ivopts', '-fno-guess-branch-probability',
           '-fprefetch-loop-arrays', '-fno-tree-coalesce-vars', '-fno-common', '-fpredictive-commoning',
           '-fno-unit-at-a-time', '-fno-cprop-registers', '-fno-early-inlining', '-fno-delete-null-pointer-checks',
           '-fselective-scheduling2', '-fno-gcse', '-fno-inline-functions-called-once', '-funswitch-loops',
           '-fno-tree-vrp', '-fno-tree-dce', '-fno-jump-tables', '-ftree-vectorize', '-fno-argument-alias',
           '-fno-schedule-insns', '-fno-branch-count-reg', '-fno-tree-switch-conversion', '-fno-auto-inc-dec',
           '-fno-crossjumping', '-fno-tree-fre', '-fno-tree-reassoc', '-fno-align-functions', '-fno-defer-pop',
           '-fno-optimize-register-move', '-fno-strict-aliasing', '-fno-rerun-cse-after-loop', '-fno-tree-ccp',
           '-fno-ipa-cp', '-fno-if-conversion2', '-fno-tree-sra', '-fno-expensive-optimizations',
           '-fno-tree-copyrename', '-fno-ipa-reference', '-fno-ipa-pure-const', '-fno-thread-jumps',
           '-fno-if-conversion', '-fno-reorder-blocks', '-falign-loops'], type=list,
                        help='A list of optimisation flags to choose from')
    parser.add_argument('--autotuning-method', default='boca', type=str,
                        help='the method for tuning optimisation flags, options: boca, rio (the default is boca)')
    args = parser.parse_args()

    return args
