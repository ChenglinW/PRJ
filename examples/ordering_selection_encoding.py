import random

class OrderingSelectionEncoder:
    def __init__(self, args_handler):
        """
        Ordering and selection encoding and decoding for algorithms
        :param args_handler: A object for handling commands and flags
        """
        self.flags = args_handler.flags

    def get_selected_flags_ordering(self, flag_order_sequence):
       """
       Get the selected flags
       :param flag_order_sequence: list[list[int, int]] a flag sequence with 0s and 1s indicate the flags enabled or not
       :return: list[String] a list of selected flags name
       """
       selected_flags = []
       for index, value in enumerate(flag_order_sequence):
              if value != 0:
                     selected_flags.insert(index, self.flags[value-1])
       return selected_flags

    def generate_flag_order_sequence_from_decimal(self, decimal_number):
        """
        generate a list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation from decimal number
        :param decimal_number: int
        :return: list[list[int, int]] A list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation
        """
        binary_sequence = bin(decimal_number).replace('0b', '')
        # Use 0s to fill up the remaining part, since the 0b has been replaced
        binary_sequence = '0' * (len(self.flags) - len(binary_sequence)) + binary_sequence
        flag_order_sequence = []
        temp = self.count_occurrences(binary_sequence, '1')
        random_index_lst = random.sample(list(range(temp)), temp)
        for binary_num in binary_sequence:
            if binary_num == '1':
                flag_order_sequence.append(random_index_lst.pop()+1)
            else:
                flag_order_sequence.append(0)

        return flag_order_sequence


    def count_occurrences(self, text, substring):
        count = 0
        index = 0

        while index < len(text):
            index = text.find(substring, index)
            if index == -1:
                break
            count += 1
            index += len(substring)

        return count