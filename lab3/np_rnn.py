import numpy as np
import scipy

from lab3.dataset import Preprocess


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / (e_x.sum(axis=1).reshape((x.shape[0], 1)) + 1e-7))


class RNN(object):
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(0, 1 / np.sqrt(min(vocab_size, hidden_size)),
                                  (vocab_size, hidden_size))  # ... input projection
        self.W = np.random.normal(0, 1 / np.sqrt(min(hidden_size, hidden_size)),
                                  (hidden_size, hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros((hidden_size, 1))  # ... input bias

        self.V = np.random.normal(0, 1 / np.sqrt(min(hidden_size, vocab_size)),
                                  (hidden_size, vocab_size))  # ... output projection
        self.c = np.zeros((vocab_size, 1))  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(
            self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current, cache = None, None

        # return the new hidden state and a tuple of values needed for the backward step

        h_current = (np.dot(h_prev, W) + np.dot(x, U)) + b.T

        h_current = np.tanh(h_current)

        cache = (h_prev, h_current, W, x)

        return h_current, cache

    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        cache = []
        h_t = []
        for i in range(self.sequence_length):
            h0, c = self.rnn_step_forward(x[:, i, :], h0, U, W, b)
            cache.append(c)
            h_t.append(h0)

        h_t = np.array(h_t).transpose((1, 0, 2))
        return h_t, cache

    # ...
    # Code is nested in class definition, indentation is not representative.

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        dh_prev, dU, dW, db = None, None, None, None

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        h_prev, h_current, W, x = cache

        dl_dh = grad_next

        dL_da = dl_dh * (1 - np.square(h_current)) / dl_dh.shape[0]

        dh_prev = np.dot(dL_da, W.T)
        dW = np.dot(h_prev.T, dL_da)
        dU = np.dot(x.T, dL_da)
        db = np.sum(dL_da, axis=0).reshape(self.b.shape)

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU_sum, dW_sum, db_sum = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        dh_prev = np.zeros_like(dh[:, 0, :])
        clip_min = -5
        clip_max = 5
        for i in range(self.sequence_length):
            dh_prev, dU, dW, db = self.rnn_step_backward(dh_prev + dh[:, -i, :], cache[-i])
            dU_sum += dU
            dW_sum += dW
            db_sum += db
            # dU_sum = np.clip(dU_sum + dU, clip_min, clip_max)
            # dW_sum = np.clip(dW_sum + dW, clip_min, clip_max)
            # db_sum = np.clip(db_sum + db, clip_min, clip_max)
        dU_sum = np.clip(dU_sum, clip_min, clip_max)
        dW_sum = np.clip(dW_sum, clip_min, clip_max)
        db_sum = np.clip(db_sum, clip_min, clip_max)
        return dU_sum, dW_sum, db_sum

    def output(self, h, V, c):
        # Calculate the output probabilities of the network
        return np.dot(h, V) + c.T

    def output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        loss, dh, dV, dc = 0, [], np.zeros_like(V), np.zeros_like(c)
        batch_size = h.shape[0]
        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        for i in range(self.sequence_length):
            h_current = h[:, i, :]
            y_current = y[:, i, :]
            o = self.output(h_current, V, c)
            p = softmax(o)

            loss += -np.sum(np.log(p) * y_current) / batch_size
            dL_do = (p - y_current) / batch_size

            dL_dV = np.dot(h_current.T, dL_do)
            dL_dc = np.sum(dL_do, axis=0).reshape(dc.shape)
            dL_dh = np.dot(dL_do, V.T)

            dV += dL_dV
            dc += dL_dc
            dh.append(dL_dh)

        return loss, np.array(dh).transpose((1, 0, 2)), dV, dc

    def update(self, dU, dW, db, dV, dc, eps=1e-7):

        # update memory matrices
        # perform the Adagrad update of parameters

        self.memory_U += np.square(dU)
        self.memory_W += np.square(dW)
        self.memory_b += np.square(db)
        self.memory_V += np.square(dV)
        self.memory_b += np.square(db)

        self.U -= self.learning_rate * dU / np.sqrt(self.memory_U + eps)
        self.W -= self.learning_rate * dW / np.sqrt(self.memory_W + eps)
        self.b -= self.learning_rate * db / np.sqrt(self.memory_b + eps)
        self.V -= self.learning_rate * dV / np.sqrt(self.memory_V + eps)
        self.c -= self.learning_rate * dc / np.sqrt(self.memory_c + eps)
        #
        # self.U -= self.learning_rate * dU
        # self.W -= self.learning_rate * dW
        # self.b -= self.learning_rate * db
        # self.V -= self.learning_rate * dV
        # self.c -= self.learning_rate * dc

    def sample(self, dataset, seed, n_sample):
        h0, seed_onehot, sample = np.zeros((self.hidden_size, 1)), dataset.one_hot(dataset.encode(seed)), []
        # inicijalizirati h0 na vektor nula
        # seed string pretvoriti u one-hot reprezentaciju ulaza
        h0 = np.zeros((1, self.hidden_size))
        s = np.array([0])
        for i, s_oh in enumerate(seed_onehot):
            h0, c = self.rnn_step_forward(s_oh, h0, self.U, self.W, self.b)
            o = self.output(h0, self.V, self.c)
            e_o = np.exp(o - np.max(o))
            p = e_o / e_o.sum()
            s = np.array([np.random.choice(np.arange(len(dataset.sorted_chars)), p=p[0])])
            sample.append(s[0])
            # print(dataset.decode(s), end='')
            # print(f'({dataset.decode([np.argmax(seed_onehot[min(i + 1, len(seed_onehot) - 1)])])}-{dataset.decode(s)})',
            #       end='')
        print()

        for i in range(n_sample):
            o = self.output(h0, self.V, self.c)
            e_o = np.exp(o - np.max(o))
            p = e_o / e_o.sum()
            s = np.array([np.random.choice(np.arange(len(dataset.sorted_chars)), p=p[0])])
            # s = np.argmax(p,axis=1)
            s_oh = dataset.one_hot(s).reshape(1, len(dataset.sorted_chars))
            h0, c = self.rnn_step_forward(s_oh, h0, self.U, self.W, self.b)
            o = self.output(h0, self.V, self.c)
            e_o = np.exp(o - np.max(o))
            p = e_o / e_o.sum()
            s = np.array([np.random.choice(np.arange(len(dataset.sorted_chars)), p=p[0])])
            # s = np.argmax(p,axis=1)
            sample.append(s[0])
        sample = np.array(sample)
        return dataset.decode(sample)

    def step(self, h0, x_oh, y_oh):
        h0, c = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)

        loss, dh, dV, dc = self.output_loss_and_grads(h0, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, c)
        self.update(dU, dW, db, dV, dc)
        return loss, h0[:, -1, :]


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-4,
                       sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)  # initialize the recurrent network

    current_epoch = 0
    batch = 1

    h0 = np.zeros((hidden_size, 1))

    average_loss = 0

    # seed = 'CORNELIUS:\nThe case was stolen?\n\n'
    # seed = 'bcdefghijk'

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()
        # print([proc.decode(xi) for xi in x])
        # print([proc.decode(yi) for yi in y])

        batch_size = x.shape[0]
        if e:
            current_epoch += 1
            h0 = np.zeros((batch_size, hidden_size))

            if batch > 1:
                print(f'Epoch {current_epoch}: {average_loss / batch}')
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh = np.array([dataset.one_hot(xi) for xi in x])
        y_oh = np.array([dataset.one_hot(yi) for yi in y])

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)

        # print(f'Batch {batch}: {loss}')
        average_loss += loss

        if batch % sample_every == 0:
            # run sampling (2.2)
            sample = rnn.sample(dataset, seed, 200)
            print(f'Batch {batch}: {average_loss / batch}')
            print(f'Sample: "{sample}"')
        batch += 1

    print(f'Epoch {current_epoch}: {average_loss / batch}')
    return rnn


def main():
    batch_size = 20
    seq_len = 100
    hidden = 200
    l_r = 5e-4
    epochs = 200
    seed = '''abcdefg'''
    proc = Preprocess(batch_size, seq_len)
    proc.preprocess('test3.txt')
    vocab = len(proc.sorted_chars)
    rnn = RNN(hidden, seq_len, vocab, l_r)
    h0 = np.zeros((batch_size, hidden))
    for e in range(epochs):
        is_new, x, y = proc.next_minibatch()
        # print([proc.decode(xi) for xi in x])
        # print([proc.decode(yi) for yi in y])
        x_oh = np.array([proc.one_hot(xi) for xi in x])
        y_oh = np.array([proc.one_hot(yi) for yi in y])
        h0, c = rnn.rnn_forward(x_oh, h0, rnn.U, rnn.W, rnn.b)

        loss, dh, dV, dc = rnn.output_loss_and_grads(h0, rnn.V, rnn.c, y_oh)
        dU, dW, db = rnn.rnn_backward(dh, c)
        # print(dU, dW, db, sep='\n')
        rnn.update(dU, dW, db, dV, dc)
        h0 = h0[:, -1, :]
        print(e + 1, loss, sep='\n')
        # sample = rnn.sample(proc, seed, 50)
        # print(f'sample:\n"{sample}"')

    for i in range(5):
        print(i)
        sample = rnn.sample(proc, seed, 50)
        print(f'Final sample:\n"{sample}"')


if __name__ == '__main__':
    batch_size = 20
    sequence_length = 50
    proc = Preprocess(batch_size, sequence_length)
    # proc.preprocess('test.txt')
    proc.preprocess('test1.txt')
    # proc.preprocess('test2.txt')
    # proc.preprocess('test3.txt')
    seed = '''EMPEROR:
There is a grave disturbance in The Force.

VADER:
What is thy bidding, My Master?

HAN:
Fine.  But they'll have to take care of themselves.

RIEEKAN:
Wait.  I'll send a patrol with you.'''
    # seed = 'bcdefghijklm'
    # seed = '''do duge mi smo plovili
    # na krilima od sjena mjeseca'''
    rnn = run_language_model(
        proc,
        850,
        sequence_length=sequence_length,
        sample_every=100,
        learning_rate=1e-1,
        hidden_size=500
    )
    for i in range(20):
        sample = rnn.sample(proc, seed, 200)
        print(f'Final sample:\n"{sample}"')
    # main()
