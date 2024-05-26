# import some packages you need here
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        self.char = sorted(list(set(text)))
        self.char2idx = {char: idx for idx, char in enumerate(self.char)}
        self.idx2char = {idx: char for idx, char in enumerate(self.char)}
        self.data = [self.char2idx[char] for char in text]

        self.seq_length = 30

    def __len__(self):

        # write your codes here
        return (len(self.data) - self.seq_length)

    def __getitem__(self, idx):

        # write your codes here
        input = torch.tensor(self.data[idx : idx + self.seq_length])
        target = torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1])

        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
    dataset = Shakespeare('shakespeare_train.txt')
    print(dataset.shape)