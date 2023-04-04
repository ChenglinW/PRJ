from config import get_args
from random_iterative import RandomIter
import os
from boca import BOCA
from bocaso import BOCASO
from input_arguments_handler import ArgsHandler

clear_exe_cmd ='ls | grep -v "\." | xargs rm'
def main(config):
    with open(config.flags_dir, 'r') as file:
        content = file.read()
        flags = content.replace('\n', ' ').split()

    if flags is not None:
        command_handler = ArgsHandler(file_cmd=config.file_cmd, baseline_cmd=config.baseline_cmd,
                                      remove_cmd=config.remove_cmd, exe_cmd=config.time_cmd,
                                      flags=flags, evaluate_times=config.evaluate_times, total_iters=config.total_iters)
        if config.autotuning_method == 'boca':

            tuning_framework = BOCA(args_handler=command_handler)

        elif config.autotuning_method == 'rio':
            tuning_framework = RandomIter(args_handler=command_handler)
        elif config.autotuning_method == 'bocaso':
            tuning_framework = BOCASO(args_handler=command_handler)
        else:
            print('There are some illegal arguments. Please use --help or -h to see what arguments can be passed')
            return
        sorted_scores, cumulative_iteration_time = tuning_framework.tuning_flags()
        best_sequence_tuple = min(sorted_scores, key=lambda key: sorted_scores[key])
        best_sequence = ' '.join(best_sequence_tuple)
        os.system(clear_exe_cmd)
        print(f'best seq: {best_sequence} time: {cumulative_iteration_time}')
    else:
        print('Please check if the flag file is correct!')






if __name__ == '__main__':
    config = get_args()
    main(config)