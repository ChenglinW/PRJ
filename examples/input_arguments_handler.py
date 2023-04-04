class ArgsHandler:
    def __init__(self,
                 file_cmd=' -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time',
                 baseline_cmd='gcc -O3 -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time',
                 remove_cmd='rm -rf *.o *.I *.s a.out',
                 exe_cmd ='./2mm_time',
                 flags=['-fno-peephole2', '-ffast-math', '-fno-schedule-insns2', '-fno-caller-saves',
                        '-funroll-all-loops',
                        '-fno-inline-small-functions', '-finline-functions', '-fno-math-errno', '-fno-tree-pre',
                        '-ftracer',
                        '-fno-reorder-functions', '-fno-dce', '-fipa-cp-clone', '-fno-move-loop-invariants',
                        '-fno-regmove',
                        '-funsafe-math-optimizations', '-fno-tree-loop-optimize', '-fno-merge-constants',
                        '-fno-omit-frame-pointer',
                        '-fno-align-labels', '-fno-tree-ter', '-fno-tree-dse', '-fwrapv', '-fgcse-after-reload',
                        '-fno-align-jumps',
                        '-fno-asynchronous-unwind-tables', '-fno-cse-follow-jumps', '-fno-ivopts',
                        '-fno-guess-branch-probability',
                        '-fprefetch-loop-arrays', '-fno-tree-coalesce-vars', '-fno-common',
                        '-fpredictive-commoning',
                        '-fno-unit-at-a-time', '-fno-cprop-registers', '-fno-early-inlining',
                        '-fno-delete-null-pointer-checks',
                        '-fselective-scheduling2', '-fno-gcse', '-fno-inline-functions-called-once',
                        '-funswitch-loops',
                        '-fno-tree-vrp', '-fno-tree-dce', '-fno-jump-tables', '-ftree-vectorize',
                        '-fno-argument-alias',
                        '-fno-schedule-insns', '-fno-branch-count-reg', '-fno-tree-switch-conversion',
                        '-fno-auto-inc-dec',
                        '-fno-crossjumping', '-fno-tree-fre', '-fno-tree-reassoc', '-fno-align-functions',
                        '-fno-defer-pop',
                        '-fno-optimize-register-move', '-fno-strict-aliasing', '-fno-rerun-cse-after-loop',
                        '-fno-tree-ccp',
                        '-fno-ipa-cp', '-fno-if-conversion2', '-fno-tree-sra', '-fno-expensive-optimizations',
                        '-fno-tree-copyrename', '-fno-ipa-reference', '-fno-ipa-pure-const',
                        '-fno-thread-jumps',
                        '-fno-if-conversion', '-fno-reorder-blocks', '-falign-loops'],
                 evaluate_times=6,
                 total_iters=60,
                 ):
        """
        Base class for all algorithms
        :param file_cmd: Command for determining what file to execute
        :param baseline_cmd: Command for executing the baseline optimisation flags
        :param remove_cmd: Command for removing the files that generated from the compilation
        :param exe_cmd: Command for executing the optimised code, so that the time can be measured
        :param flags: A list of optimisation flags to choose from
        """
        self.file_cmd = file_cmd
        self.baseline_cmd = baseline_cmd
        self.remove_cmd = remove_cmd
        self.exe_cmd = exe_cmd
        self.flags = flags
        self.evaluate_times=evaluate_times
        self.total_iters=total_iters
