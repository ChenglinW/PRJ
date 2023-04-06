import os, sys
import time
import numpy as np
import random
from input_arguments_handler import ArgsHandler

class EvaluationComponent:
    def __init__(self,
                 args_handler,):
        """
        Base class for all algorithms
        :param args_handler: a command handler stores all needed commands
        """
        self.file_cmd = args_handler.file_cmd
        self.baseline_cmd = args_handler.baseline_cmd
        self.remove_cmd = args_handler.remove_cmd
        self.exe_cmd = args_handler.exe_cmd
        self.evaluate_times = args_handler.evaluate_times
        self.total_iters = args_handler.total_iters


    def get_evaluation_score(self, selected_flags_names):
        """
        Get the evaluation score for a unit of optimisation flags
        :param selected_flags_names: list[string] the name of the segt of optimisation flags that is going to be executed
        :return: float the median of total evaluation scores
        """
        try:
            speedups = []
            step = 0
            while len(speedups) < self.evaluate_times:
                step += 1
                os.system(self.remove_cmd)
                compile_cmd = 'gcc -O2 ' + ' '.join(selected_flags_names) + self.file_cmd
                print(compile_cmd)
                compile_status = os.system(compile_cmd)
                if compile_status != 0:
                    print(f"Something went wrong! Please check program {self.file_cmd}.")
                    sys.exit(1)
                    break
                begin = time.time()
                print(self.exe_cmd)
                exe_status = os.system(self.exe_cmd)
                if exe_status != 0:
                    print(f"Something went wrong! Please check program {self.exe_cmd}.")
                    sys.exit(1)
                    break
                print(exe_status)
                end = time.time()
                chosen_sequence_time = end - begin
                os.system(self.remove_cmd)
                os.system(self.baseline_cmd)

                begin = time.time()
                os.system(self.exe_cmd)
                end = time.time()
                baseline_time = end - begin

                print('baseline:' + str(baseline_time) + 'selected flags:' + str(chosen_sequence_time) + ' val:' + str(baseline_time / chosen_sequence_time))
                speedups.append(baseline_time / chosen_sequence_time)

            print(speedups)
        except Exception as e:
            print(f"An error occurred: {e}")
        return -np.median(speedups)



    def tuning_flags(self):
        """
        The main function for optimisation flags autotuning
        :return:
        """
        raise NotImplementedError("Please Implement this method")
