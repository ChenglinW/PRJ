from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import time
import random
import math
import numpy as np
from base_autotuning_structure import BaseAlgorithm
import wandb

class BOCAT(BaseAlgorithm):
    """
        the bayesian optimisation with random forest
    """
    def __init__(self, model = 1,
                 fnum = 8,
                 decay = 0.5,
                 scale = 10.0,
                 offset = 20.0,
                 init_sample_size = 2,
                 rnum = 8,
                 *args, **kwargs):
        self.md = model
        self.fnum = fnum # number of features
        self.decay = decay
        self.scale = scale
        self.offset = offset
        self.init_sample_size = init_sample_size
        self.rnum = rnum
        super().__init__(*args, **kwargs)


    def get_EI(self, prediction, eta):
        """
        compute and return the expected improvement

        :param prediction: list[ndarray[float, float]] A list of prediction from different decision tree in random forest
        :param eta: float The best evaluation score
        :return: ndarray[float, float] The ndarray of two expected improvement
        """

        prediction = np.array(prediction).transpose(1, 0)
        mean = np.mean(prediction, axis=1)
        sd = np.std(prediction, axis=1)

        def calculate_f():
            z = (eta - mean) / sd # z represents standardised value
            return (eta - mean) * norm.cdf(z) + sd * norm.pdf(z)

        if np.any(sd == 0.0):
            s_copy = np.copy(sd)
            sd[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        return f

    def change_selected_flags(self, selected_flags, flag_choice_mappings):
        """

        :param selected_flags: list[int] A list of selected flags index in the whole flag options
        :param flag_choice_mappings: list[list(int, int)] A list of tuples of index of flags-to-be-changed and enable & disable choice
        :return: list[int] A list of 0s and 1s binary flag sequence
        """
        sequence = [0] * len(self.flags)
        for flag_index in selected_flags:
            sequence[flag_index] = 1

        for flag_choice_pair in flag_choice_mappings:
            sequence[flag_choice_pair[0]] = flag_choice_pair[1] #set the flag to its choice
        return sequence

    def get_important_features(self, model):
        """
        Get the feature value array
        :param model: The random forest model used to get the features
        :return: 2darray: a 2 dimensional numpy array of importance value for each flags
        """
        features = model.feature_importances_

        return features

    def get_all_possible_flag_choice_mappings(self, selected_features):
        """
        To get all possible flag mapping to the enable&disable choice
        :param selected_features: list[list[int, float]] A list of flag index and gini importance value
        :return: A list of flags index and its enable&disable choice
        """
        flag_choice_mappings = []
        for i in range(2 ** self.fnum):
            comb = bin(i).replace('0b', '')
            comb = '0' * (self.fnum - len(comb)) + comb
            current_choices = []
            for k, s in enumerate(comb):
                if s == '1':
                    current_choices.append((selected_features[k][0], 1))
                else:
                    current_choices.append((selected_features[k][0], 0))
            flag_choice_mappings.append(current_choices)

        return flag_choice_mappings

    def random_search(self, model, eta, rnum):
        features = self.get_important_features(model)
        print("features")
        print(features)

        start_time = time.time()
        sorted_features = [[i, x] for i, x in enumerate(features)]
        selected_features = sorted(sorted_features, key=lambda x: x[1], reverse=True)[:self.fnum]
        feature_indices = [x[0] for x in sorted_features]

        # listing out all the combinations of the selected feature
        flag_choice_mappings = self.get_all_possible_flag_choice_mappings(selected_features)

        # all possible flag sequences
        sequences = [] # list[list[int]] List of all 0s and 1s binary flag sequences
        for flag_choice_mapping in flag_choice_mappings:
            for j in range(int(rnum) + 1):
                selected_feature_indices = random.sample(feature_indices, random.randint(0, len(feature_indices)))
                binary_flag_sequence = self.change_selected_flags(selected_feature_indices, flag_choice_mapping)
                sequences.append(binary_flag_sequence)

        print(f'Length of all flags sequences to be estimated: {len(sequences)}')
        expected_improvement = self.get_prediction_ei(model, sequences, eta)
        sequence_ei_mappings = [[i, a] for a, i in zip(expected_improvement, sequences)]
        total_time = time.time() - start_time
        print(f'The time for adding the random flags: {total_time}')
        return sequence_ei_mappings

    def get_prediction_ei(self, model, sequences, eta):
        prediction = []
        estimators = model.estimators_
        for estimator in estimators:
            prediction.append(estimator.predict(np.array(sequences)))
        expected_improvement = self.get_EI(prediction, eta)

        return expected_improvement

    def train_random_forest(self, flag_sequences, evaluation_scores):
        """
        Train the random forest
        :param flag_sequences: list[list[int]] A list of binary flag sequences
        :param evaluation_scores: list[float] A list of evaluation scores for the flag sequences
        :return: The random forest regression model
        """
        model = RandomForestRegressor()
        model.fit(np.array(flag_sequences), np.array(evaluation_scores))

        return model

    def get_best_estimated_non_repetitive_flag_sequence_and_ei(self, flag_sequences, evaluation_scores, eta, rnum):
        """
        Get the best estimated flag sequence and its expected improvement
        :param flag_sequences: list[list[int]] A list of binary flag sequences
        :param evaluation_scores: list[float] A list of evaluation scores for the flag sequences
        :param eta: float The best evaluation score
        :param rnum: float random selection size
        :return: the pair of the best non-repetitive estimated flag sequence and expected improvement
        """

        model = self.train_random_forest(flag_sequences, evaluation_scores)

        sequence_ei_mappings = self.random_search(model, eta, rnum)
        sorted_sequence_ei_mappings = sorted(sequence_ei_mappings, key=lambda x: x[1], reverse=True)

        best_estimated_flag_sequence_and_ei = ()
        start_time = time.time()
        for sequence_ei_mapping in sorted_sequence_ei_mappings:
            if sequence_ei_mapping[0] not in flag_sequences:
                best_estimated_flag_sequence_and_ei = sequence_ei_mapping[0], sequence_ei_mapping[1]
                break
        print(f'Time for finding non-repetitive {time.time() - start_time}')
        return best_estimated_flag_sequence_and_ei

    def tuning_flags(self):
        wandb.login()
        wandb.init(
            project='Compiler Autotuning',
            name= f'BOCA {self.time_cmd} Iterations: {self.total_iters} Number of flags: {len(self.flags)}',
            config={
                'Program_name': self.time_cmd,
                'Iterations': self.total_iters,
                'Number of Flags': len(self.flags),
            },
        )
        binary_flag_sequences = []
        cumulative_iteration_time = []
        evaluation_scores = []
        actual_flag_sequences = []
        init_rnum = 2 ** self.rnum
        start_time = time.time()
        sigma = -self.scale ** 2 / (2 * math.log(self.decay))

        while len(binary_flag_sequences) < self.init_sample_size:
            random_decimal_number = random.randint(0, 2 ** len(self.flags))
            binary_flag_sequence = self.generate_flag_sequence_from_decimal(random_decimal_number)
            if binary_flag_sequence not in binary_flag_sequences:
                binary_flag_sequences.append(binary_flag_sequence)
                actual_flag_sequence = self.get_selected_flags(binary_flag_sequence)
                evaluation_scores.append(self.get_evaluation_score(actual_flag_sequence))
                cumulative_iteration_time.append(time.time() - start_time)

        current_step = 0
        result = 100000000

        for score in evaluation_scores:
            if score < result:
                result = score

        while self.init_sample_size+current_step < self.total_iters:
            current_step += 1
            rnum = init_rnum * math.exp(-max(0, len(binary_flag_sequences) - self.offset) ** 2 / (2 * sigma ** 2))
            best_solution, return_nd_independent = self.get_best_estimated_non_repetitive_flag_sequence_and_ei(binary_flag_sequences, evaluation_scores, result, rnum)
            binary_flag_sequences.append(best_solution)
            cumulative_iteration_time.append(time.time() - start_time)
            actual_flag_sequence = self.get_selected_flags(best_solution)
            actual_flag_sequences.append(actual_flag_sequence)
            best_result = self.get_evaluation_score(actual_flag_sequence)
            evaluation_scores.append(best_result)
            wandb.log({"evaluation score": best_result})
            if best_result < result:
                result = best_result

            # print('current best_solution')
            # print(best_solution)
        mapped_flags_scores = {tuple(actual_flag_sequences[index]): evaluation_scores[index] for index in
                               range(len(actual_flag_sequences))}
        print(f'Total running time {time.time() - start_time}')
        return mapped_flags_scores, cumulative_iteration_time
