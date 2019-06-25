import numpy as np
from collections import Counter


class Preprocess(object):
    def __init__(self, batch_size: int, sequence_length: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.batches = None
        self.minibatches_x, self.minibatches_y = None, None

    # ...

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters

        counter = Counter()
        for c in data:
            counter[c] = counter[c] + 1

        self.sorted_chars = sorted(counter.keys(), key=lambda c: -counter[c])

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        decoded_sequence = ""
        for id in encoded_sequence:
            decoded_sequence += self.id2char[id]
        return decoded_sequence

    def create_plenty_minibatches(self):
        self.x = self.x[:10000]
        x = self.x[:-1]
        y = self.x[1:]
        self.num_batches = int((len(x) - self.sequence_length + 1) / self.batch_size)  # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################

        batches_x = []
        batches_y = []
        for i in range(self.num_batches):
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                batch_x.append(
                    x[
                    self.batch_size * i + j
                    :
                    self.batch_size * i + j + self.sequence_length
                    ]
                )
                batch_y.append(
                    y[
                    self.batch_size * i + j
                    :
                    self.batch_size * i + j + self.sequence_length
                    ]
                )
            batches_x.append(batch_x)
            batches_y.append(batch_y)
        self.batches = (np.array(batches_x), np.array(batches_y))
        return self.batches

    def create_minibatches(self):
        self.x = self.x[:20000]
        x = self.x[:-1]
        y = self.x[1:]
        self.num_batches = int(
            len(x) / (self.batch_size * self.sequence_length))  # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################

        batches_x = []
        batches_y = []
        for i in range(self.num_batches):
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                batch_x.append(
                    x[
                    self.batch_size * i * self.sequence_length + j * self.sequence_length
                    :
                    self.batch_size * i * self.sequence_length + (j + 1) * self.sequence_length
                    ]
                )
                batch_y.append(
                    y[
                    self.batch_size * i * self.sequence_length + j * self.sequence_length
                    :
                    self.batch_size * i * self.sequence_length + (j + 1) * self.sequence_length
                    ]
                )
            batches_x.append(batch_x)
            batches_y.append(batch_y)
            # batches_x = [batch_x]
            # batches_y = [batch_y]
            # break
        self.batches = (np.array(batches_x), np.array(batches_y))
        return self.batches

    def next_minibatch(self):
        # ...

        batch_x, batch_y = None, None
        new_epoch = False
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        if self.batches is None:
            self.create_minibatches()
        if self.minibatches_x is None or self.minibatches_y is None:
            self.minibatches_x, self.minibatches_y = iter(self.batches[0]), iter(self.batches[1])
            new_epoch = True
        try:
            batch_x, batch_y = self.minibatches_x.__next__(), self.minibatches_y.__next__()
        except StopIteration:
            self.minibatches_x, self.minibatches_y = iter(self.batches[0]), iter(self.batches[1])
            new_epoch = True
            batch_x, batch_y = self.minibatches_x.__next__(), self.minibatches_y.__next__()
        return new_epoch, batch_x, batch_y

    def one_hot(self, x):
        s = len(self.sorted_chars)
        oh = np.zeros((x.size, s))
        oh[np.arange(x.size), x] = 1
        return oh


if __name__ == '__main__':
    processor = Preprocess(2, 2)
    processor.preprocess('test.txt')
    print(processor.x)
    print(processor.one_hot(processor.x))
    print(processor.encode('ababcad'))
    print("'", processor.decode([1, 2, 3, 4, 0, 0, 1]), "'", sep='')
    print(processor.create_minibatches())
    for i in range(9):
        print(*processor.next_minibatch(), sep='\n')
