import time
import random
from evaluation_component import EvaluationComponent
from ordering_selection_encoding import OrderingSelectionEncoder

class RandomIter(EvaluationComponent, OrderingSelectionEncoder):
    """
    Random iterative optimisation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        OrderingSelectionEncoder.__init__(self, *args, **kwargs)
    def tuning_flags(self):
        """
        Autotuning the optimisation flags by using random iterative algorithm
        :return: dict{tuple(str): float} A list of tuples of binary flag sequence and its evaluation score, and the list of time for each iteration
        """
        binary_flag_sequences = []
        current_random_numbers = [] # sotre data to avoid the duplication for random number
        evaluation_scores = []
        cumulative_iteration_time = []
        actual_flag_sequences = []
        start_time = time.time()
        while len(current_random_numbers) < self.total_iters:
            random_decimal_number = random.randint(0, 2 ** len(self.flags))
            if random_decimal_number not in current_random_numbers:
                current_random_numbers.append(random_decimal_number)
                binary_flag_sequence = self.generate_flag_order_sequence_from_decimal(random_decimal_number)
                binary_flag_sequences.append(binary_flag_sequence)
                actual_flag_sequence = self.get_selected_flags_ordering(binary_flag_sequence)
                actual_flag_sequences.append(actual_flag_sequence)
                evaluation_score = self.get_evaluation_score(actual_flag_sequence)
                evaluation_scores.append(evaluation_score)
                iter_time = time.time() - start_time
                cumulative_iteration_time.append(iter_time)
        print(f'Total running time {time.time() - start_time}')
        mapped_flags_scores = {tuple(actual_flag_sequences[index]) : evaluation_scores[index] for index in range(len(actual_flag_sequences))}
      #  sorted_scores = [mapped_flag_score[1] for mapped_flag_score in mapped_flags_scores]
      #  return binary_flag_sequences, mapped_flags_scores, cumulative_iteration_time
        return mapped_flags_scores, cumulative_iteration_time
