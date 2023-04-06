class SelectionEncoder:
    def __init__(self, args_handler):
        """
        selection encoding and decoding for algorithms
        :param args_handler: A object for handling commands and flags
        """
        self.flags = args_handler.flags

    def get_selected_flags(self, flag_sequence):
       """
       Get the selected flags
       :param flag_sequence: list[int] a flag sequence with 0s and 1s indicate the flags enabled or not
       :return: list[String] a list of selected flags name
       """
       selected_flags = []
       for index, value in enumerate(flag_sequence):
              if value == 1:
                     selected_flags.append(self.flags[index])
       return selected_flags

    def generate_flag_sequence_from_decimal(self, decimal_number):
        """
        generate a list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation from decimal number
        :param decimal_number:
        :return: list[int] A list of 0s and 1s that stands for the flags sequence for the actual flag sequence generation
        """
        binary_sequence = bin(decimal_number).replace('0b', '')
        # Use 0s to fill up the remaining part, since the 0b has been replaced
        binary_sequence = '0' * (len(self.flags) - len(binary_sequence)) + binary_sequence
        sequence_lst = []
        for binary_num in binary_sequence:
            if binary_num == '1':
                sequence_lst.append(1)
            else:
                sequence_lst.append(0)

        return sequence_lst