import os, sys
import time
import numpy as np

class BaseAlgorithm:
    def __init__(self,
                 cmd2=' -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time',
                 cmd3='gcc -O2 -funswitch-loops -ftree-vectorize -fpredictive-commoning -fipa-cp-clone -finline-functions -fgcse-after-reload -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time',
                 cmd4='rm -rf *.o *.I *.s a.out',
                 cmd5='./2mm_time',
                 flags=['-fno-peephole2', '-ffast-math', '-fno-schedule-insns2', '-fno-caller-saves', '-funroll-all-loops',
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
           '-fno-if-conversion', '-fno-reorder-blocks', '-falign-loops'],
                 evaluate_times = 6,
                 total_iters = 60):
        """
        Base class for all algorithms
        :param cmd2: Command for determining what file to execute
        :param cmd3: Command for determining the baseline optimisation flags
        :param cmd4: Command for removing the files that generated from the benchmark
        :param cmd5: Command for getting the execution time of the target benchmark file
        :param flags: A list of optimisation flags to choose from
        :param evaluate_times: the times for evaluate a flag sequence
        :param total_iters: the iteration times for the algorithm
        """
        self.file_cmd = cmd2
        self.baseline_cmd = cmd3
        self.remove_cmd = cmd4
        self.time_cmd = cmd5
        self.flags = flags
        self.evaluate_times = evaluate_times
        self.total_iters = total_iters


    def get_selected_flags(self, flag_sequence):
       """
       Get the selected flags
       :param flag_sequence: list[int] a flag sequence with 0s and 1s indicate the flags enabled or not
       :return: list[String] a list of selected flags name
       """
       selected_flags = []
       for index, value in enumerate(flag_sequence):
              if value == 1:
                     selected_flags.append(self.flags[index])
       return selected_flags

    def get_evaluation_score(self, selected_flags_names):
        """
        Get the evaluation score for a unit of optimisation flags
        :param selected_flags_names: list[string] the name of the segt of optimisation flags that is going to be executed
        :return: float the median of total evaluation scores
        """
        speedups = []
        step = 0
        while len(speedups) < self.evaluate_times:
            step += 1
            if step > 10:
                print('failed configuration!')
                sys.exit(0)
            os.system(self.remove_cmd)
            print('gcc -O2 ' + ' '.join(selected_flags_names) + self.file_cmd)
            os.system('gcc -O2 ' + ' '.join(selected_flags_names) + self.file_cmd)
            begin = time.time()
            print(self.time_cmd)
            ret = os.system(self.time_cmd)
            if ret != 0:
                continue
            print(ret)
            end = time.time()
            de = end - begin
            os.system(self.remove_cmd)
            os.system(self.baseline_cmd)

            begin = time.time()
            os.system(self.time_cmd)
            end = time.time()
            nu = end - begin

            print('baseline:' + str(nu) + 'selected flags:' + str(de) + ' val:' + str(nu / de))
            speedups.append(nu / de)

        print(speedups)
        return -np.median(speedups)

    def generate_flag_sequence_from_decimal(self, decimal_number):
        """
        generate a list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation from decimal number
        :param decimal_number:
        :return: list[int] A list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation
        """
        binary_sequence = bin(decimal_number).replace('0b', '')
        # Use 0s to fill up the remaining part, since the 0b has been replaced
        binary_sequence = '0' * (len(self.flags) - len(binary_sequence)) + binary_sequence
        sequence_lst = []
        for binary_num in binary_sequence:
            if binary_num == '1':
                sequence_lst.append(1)
            else:
                sequence_lst.append(0)

        return sequence_lst
    def tuning_flags(self):
        """
        The main function for optimisation flags autotuning
        :return:
        """
        raise NotImplementedError("Please Implement this method")
