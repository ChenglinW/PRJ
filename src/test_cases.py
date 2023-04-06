import unittest

from bocaso import BOCASO
from boca import BOCA
import numpy as np
from io import StringIO
import sys
from random_iterative import RandomIter
from ordering_selection_encoding import OrderingSelectionEncoder
from sklearn.model_selection import train_test_split
from selection_encoding import SelectionEncoder
from evaluation_component import EvaluationComponent
from input_arguments_handler import ArgsHandler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import random


class TestAutotuningSystem(unittest.TestCase):

    def test_selection_flag_decoding(self):
        args_handler = ArgsHandler()  # use its default values
        selection_encoder = SelectionEncoder(args_handler)
        dummy_binary_flag_choices = [1] + [0]*(len(selection_encoder.flags)-1)
        self.assertEqual(selection_encoder.get_selected_flags(dummy_binary_flag_choices), ['-fno-peephole2'])

    def test_ordering_selection_flag_decoding(self):
        arg_handler = ArgsHandler()
        ordering_selection_encoder = OrderingSelectionEncoder(arg_handler)
        dummy_binary_flag_choices = [2, 1] + [0] * (len(ordering_selection_encoder.flags) - 2)
        self.assertEqual(ordering_selection_encoder.get_selected_flags_ordering(dummy_binary_flag_choices), ['-ffast-math', '-fno-peephole2'])

    def test_selection_flag_encoding(self):
        args_handler = ArgsHandler()  # use its default values
        selection_encoder = SelectionEncoder(args_handler)
        one_flag = selection_encoder.generate_flag_sequence_from_decimal(1)
        self.assertTrue(1 in one_flag)
        self.assertTrue(0 in one_flag)
        self.assertFalse(2 in one_flag)
        two_flags = selection_encoder.generate_flag_sequence_from_decimal(2)
        self.assertTrue(1 in two_flags)
        self.assertTrue(0 in two_flags)
        self.assertFalse(2 in two_flags)
        self.assertEqual(one_flag, [0]*(len(selection_encoder.flags)-1) + [1])

    def test_ordering_selection_flag_encoding(self):
        args_handler = ArgsHandler()  # use its default values
        ordering_selection_encoder = OrderingSelectionEncoder(args_handler)
        one_flag = ordering_selection_encoder.generate_flag_order_sequence_from_decimal(1)
        self.assertTrue(1 in one_flag)
        self.assertTrue(0 in one_flag)
        self.assertFalse(2 in one_flag)
        two_flags = ordering_selection_encoder.generate_flag_order_sequence_from_decimal(3)
        self.assertTrue(1 in two_flags)
        self.assertTrue(0 in two_flags)
        self.assertTrue(2 in two_flags)

    def test_boca_ei_calculation(self):
        args_handler = ArgsHandler()
        boca = BOCA(args_handler=args_handler)
        prediction = [
            np.array([2.0, 3.0]),
            np.array([2.5, 3.5]),
            np.array([1.5, 2.5])
        ]
        eta = 1.0
        ei = boca.get_EI(prediction, eta)

        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, (2,))
        self.assertTrue(np.all(ei >= 0))

        prediction_same = [
            np.array([2.0, 3.0]),
            np.array([2.0, 3.0]),
            np.array([2.0, 3.0])
        ]
        ei_zero = boca.get_EI(prediction_same, eta)
        self.assertEqual(ei_zero[0], 0)
        self.assertEqual(ei_zero[1], 0)


    def test_bocaso_ei_calculation(self):
        args_handler = ArgsHandler()
        bocaso = BOCASO(args_handler=args_handler)
        prediction = [
            np.array([2.0, 3.0]),
            np.array([2.5, 3.5]),
            np.array([1.5, 2.5])
        ]
        eta = 1.0
        ei = bocaso.get_EI(prediction, eta)

        self.assertIsInstance(ei, np.ndarray)
        self.assertEqual(ei.shape, (2,))
        self.assertTrue(np.all(ei >= 0))

        prediction_same = [
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0])
        ]
        ei_zero = bocaso.get_EI(prediction_same, eta)
        self.assertEqual(ei_zero[0], 0)
        self.assertEqual(ei_zero[1], 0)

    def test_boca_random_forest_training_prediction_boca(self):
        # some code modified from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/tests/test_forest.py
        args_handler = ArgsHandler()
        boca = BOCA(args_handler=args_handler)
        # Generate dummy regression dataset
        X, y = make_regression(n_samples=200, n_features=2, noise=0.001)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        X_train = list(X_train)
        y_train = list (y_train)

        model  = boca.train_random_forest(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        self.assertLessEqual(mse, 3000)

    def test_bocaso_random_forest_training_prediction_bocaso(self):
        # some code modified from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/tests/test_forest.py
        args_handler = ArgsHandler()
        bocaso = BOCASO(args_handler=args_handler)
        # Generate dummy regression dataset
        X, y = make_regression(n_samples=200, n_features=2, noise=0.001)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        X_train = list(X_train)
        y_train = list (y_train)

        model  = bocaso.train_random_forest(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        self.assertLessEqual(mse, 3000)


    def test_feature_importance_boca(self):
        args_handler = ArgsHandler()
        boca = BOCA(args_handler=args_handler)
        # Generate dummy regression dataset
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        X_train = list(X_train)
        y_train = list(y_train)

        model = boca.train_random_forest(X_train, y_train)

        features = boca.get_important_features(model)
        self.assertFalse(features is None)

    def test_feature_importance_bocaso(self):
        args_handler = ArgsHandler()
        bocaso = BOCASO(args_handler=args_handler)
        # Generate dummy regression dataset
        X, y = make_regression(n_samples=200, n_features=10, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        X_train = list(X_train)
        y_train = list(y_train)

        model = bocaso.train_random_forest(X_train, y_train)

        features = bocaso.get_important_features(model)
        self.assertFalse(features is None)

    def test_permutation_generation(self):
        args_handler = ArgsHandler()
        bocaso = BOCASO(args_handler=args_handler)
        permutation = bocaso.get_random_selected_permuted_sequence([0,2,7,8,9], [[0,0], [2,0], [7,0]])
        self.assertTrue(permutation is not None)
        self.assertTrue(1 in permutation)
        self.assertTrue(2 in permutation)
        self.assertFalse(3 in permutation)
        self.assertTrue(permutation.index(1) - permutation.index(2) < 3)


    def test_if_evaluation_component_can_execute_code(self):
        args_handler = ArgsHandler(file_cmd=' -o hello_world Test_C_Programs/hello_world.c',
                                   baseline_cmd='gcc -o hello_world Test_C_Programs/hello_world.c',
                                   exe_cmd='./hello_world')
        eval_component = EvaluationComponent(args_handler)

        score = eval_component.get_evaluation_score(['-ffast-math'])
        self.assertFalse(score is None)

    def test_if_evaluation_component_can_handle_buggy_code(self):
        with self.assertRaises(SystemExit) as sysexit:
            args_handler = ArgsHandler(file_cmd=' -o buggy_hello_world Test_C_Programs/buggy_hello_world.c',
                                       baseline_cmd='gcc -o buggy_hello_world Test_C_Programs/buggy_hello_world.c',
                                       exe_cmd='./buggy_hello_world')
            eval_component = EvaluationComponent(args_handler)
            eval_component.get_evaluation_score(['-ffast-math'])

        self.assertEqual(sysexit.exception.code, 1)

if __name__ == '__main__':
    unittest.main()