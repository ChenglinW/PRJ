import time
import random
from base_autotuning_structure import BaseAlgorithm

class RandomIter(BaseAlgorithm):
    """
    Random iterative optimisation
    """
    def __init__(self, cmd2, cmd3, cmd4, cmd5):
        super().__init__(cmd2, cmd3, cmd4, cmd5)

    def tuning_flags(self):
        """
        Autotuning the optimisation flags by using random iterative algorithm
        :return: tuple[list[tuple[string, float]], list[float]] A list of tuples of binary flag sequence and its evaluation score, and the list of time for each iteration
        """
        binary_flag_sequences = []
        current_random_numbers = [] # some data to avoid the duplication for random number
        evaluation_scores = []
        cumulative_iteration_time = []
        start_time = time.time()
        while len(current_random_numbers) < self.total_iters:
            random_decimal_number = random.randint(0, 2 ** len(self.flags))
            if random_decimal_number not in current_random_numbers:
                current_random_numbers.append(random_decimal_number)
                binary_flag_sequence = self.generate_flag_sequence_from_decimal(random_decimal_number)
                binary_flag_sequences.append(binary_flag_sequence)
                actual_flag_sequence = self.get_selected_flags(binary_flag_sequence)
                evaluation_scores.append(self.get_evaluation_score(actual_flag_sequence))
                cumulative_iteration_time.append(time.time() - start_time)
        print(f'Total running time {time.time() - start_time}')
        mapped_flags_scores = [[flag_sequence, evaluation_scores[index]] for index, flag_sequence in enumerate(binary_flag_sequences)]
        sorted_scores = [mapped_flag_score[1] for mapped_flag_score in mapped_flags_scores]
        return sorted_scores, cumulative_iteration_time
