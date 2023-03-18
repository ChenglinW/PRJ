import os, sys
import random
import math
import numpy as np
import numpy as np
import time

init_sample_size = 2
iters = 10
begin2end = 5

cmd2 = ' -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time'
cmd3 = 'gcc -O2 -funswitch-loops -ftree-vectorize -fpredictive-commoning -fipa-cp-clone -finline-functions -fgcse-after-reload -I ../polybench/utilities -I ../polybench/linear-algebra/kernels/2mm ../polybench/utilities/polybench.c ../polybench/linear-algebra/kernels/2mm/2mm.c -lm -DPOLYBENCH_TIME -o 2mm_time'
cmd4 = 'rm -rf *.o *.I *.s a.out'
cmd5 = './2mm_time'

options = ['-fno-peephole2', '-ffast-math', '-fno-schedule-insns2', '-fno-caller-saves', '-funroll-all-loops',
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
           '-fno-if-conversion', '-fno-reorder-blocks', '-falign-loops']

def generate_opts(independent):
    result = []
    for k, s in enumerate(independent):
        if s == 1:
            result.append(options[k])
    independent = result

    return independent


def change_opt_ordering(opts_seq):
    random.shuffle(opts_seq)


#evaluate it 6 times and find the median of its running time
def get_objective_score(independent):
   # independent = generate_opts(independent)

    speedups = []
    step = 0
   # TODO: set the number to constant
    while (len(speedups) < 3):
        step += 1
        if step > 10:
            print('failed configuration!')
            sys.exit(0)
        os.system(cmd4)
        print('gcc -O2 ' + ' '.join(independent) + cmd2)
        os.system('gcc -O2 ' + ' '.join(independent) + cmd2)
        begin = time.time()
        print(cmd5)
        ret = os.system(cmd5)
        if ret != 0:
            continue
        print(ret)
        end = time.time()
        de = end - begin
        os.system(cmd4)
        os.system(cmd3)

        begin = time.time()
        os.system(cmd5)
        end = time.time()
        nu = end - begin

        print('nu:' + str(nu) + ' de:' + str(de) + ' val:' + str(nu / de))
        speedups.append(nu / de)

    print(speedups)
    return -np.median(speedups)

def main():
    training_indep = []
    indep = []
    dep = []
    opts_seq =[]
    ts = []
    b = time.time()
    x = random.randint(0, 2 ** len(options))
    while len(training_indep) < iters:
        training_indep.append(x)
        comb = bin(x).replace('0b', '')
        comb = '0' * (len(options) - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        indep.append(conf)
        opts_seq = generate_opts(conf)
        change_opt_ordering(opts_seq)
        dep.append(get_objective_score(opts_seq))
        ts.append(time.time() - b)
    print('time:' + str(time.time() - b))
    #     if x not in training_indep:
    #         training_indep.append(x)
    #         comb = bin(x).replace('0b', '')
    #         comb = '0' * (len(options) - len(comb)) + comb
    #         conf = []
    #         for flag in comb:
    #             if flag == '1':
    #                 conf.append(1)
    #             else:
    #                 conf.append(0)
    #         indep.append(conf)
    #         dep.append(get_objective_score(conf))
    #         ts.append(time.time() - b)
    # training_dep = [get_objective_score(r) for r in training_indep]
    return indep, ts

if __name__ == '__main__':
    stats = []
    times = []

    for i in range(begin2end):
        dep, ts = main()
        stats.append(dep)
        times.append(ts)

    vals = []
    for j, v_tmp in enumerate(stats):
        max_s = 0
        for i, v in enumerate(v_tmp):
            max_s = min(max_s, v)
            v_tmp[i] = max_s

    print(stats)

    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(times[j][i])
        vals.append(-np.mean(tmp))

    print(vals)

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(-np.mean(tmp))

    print(vals)

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(-np.std(tmp))

    print(vals)