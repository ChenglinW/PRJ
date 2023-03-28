from config import get_args
from rand_iter_optim import RandomIter
from bo_rf import BOCA
import wandb


def main(config):
    if config.autotuning_method == 'boca':
        tuning_framework = BOCA(cmd2=config.file_cmd, cmd3=config.baseline_cmd,
                                cmd4=config.remove_cmd, cmd5=config.time_cmd,
                                flags=config.flags, evaluate_times=config.evaluate_times, total_iters=config.total_iters)

    elif config.autotuning_method == 'rio':
        tuning_framework = RandomIter(cmd2=config.file_cmd, cmd3=config.baseline_cmd,
                                      cmd4=config.remove_cmd, cmd5=config.time_cmd,
                                      flags=config.flags, evaluate_times=config.evaluate_times, total_iters=config.total_iters)
    else:
        print('There are some illegal arguments. Please use --help or -h to see what arguments can be passed')
        return
    sorted_scores, cumulative_iteration_time = tuning_framework.tuning_flags()
    best_sequence = min(sorted_scores, key=lambda key: sorted_scores[key])
    print(f'best seq: {best_sequence} time: {cumulative_iteration_time}')




if __name__ == '__main__':
    config = get_args()
    main(config)