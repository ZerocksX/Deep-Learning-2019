import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
n_samples = mnist.train.num_examples

learning_rate = 0.001
batch_size = 100

n_hidden_recog_1 = 200  # 1 sloj enkodera
n_hidden_recog_2 = 200  # 2 sloj enkodera
n_hidden_gener_1 = 200  # 1 sloj dekodera
n_hidden_gener_2 = 200  # 2 sloj dekodera
n_z = 2  # broj skrivenih varijabli
n_input = 784  # MNIST data input (img shape: 28*28)
in_shape = (28, 28)


def get_canvas(Z, ind, nx, ny, in_shape, batch_size, sess):
    """Crtanje rekonstrukcija na odgovarajućim pozicijama u 2D prostoru skrivenih varijabli
    Z -- skriveni vektori raspoređeni u gridu oko ishodišta
    ind -- indeksi za rezanje Z-a na batch_size blokove za slanje u graf -zbog problema sa random generatorom
    nx -- raspon grida po x osi - skrivena varijabla z0
    ny -- raspon grida po y osi - skrivena varijabla z1
    in_shape -- dimenzije jedne rekonstrukcije i.e. ulazne sličice
    batch_size -- veličina minibatcha na koji je graf naviknut
    sess -- session grafa mreže
    """
    # get reconstructions for visualiations
    X = np.empty((0, in_shape[0] * in_shape[1]))  # empty array for concatenation
    # split hidden vectors into minibatches of batch_size due to TF random generator limitation
    for batch in np.array_split(Z, ind):
        # fill up last batch to full batch_size if neccessary
        # this addition will not be visualized, but is here to avoid TF error
        if batch.shape[0] < batch_size:
            batch = np.concatenate((batch, np.zeros((batch_size - batch.shape[0], batch.shape[1]))), 0)
        # get batch_size reconstructions and add them to array of previous reconstructions
        X = np.vstack((X, sess.run(x_reconstr_mean_out, feed_dict={z: batch})))
    # make canvas with reconstruction tiles arranged by the hidden state coordinates of each reconstruction
    # this is achieved for all reconstructions by clever use of reshape, swapaxes and axis inversion
    return (X[0:nx * ny, :].reshape((nx * ny, in_shape[0], in_shape[1])).swapaxes(0, 1)
            .reshape((in_shape[0], ny, nx * in_shape[1])).swapaxes(0, 1)[::-1, :, :]
            .reshape((ny * in_shape[0], nx * in_shape[1])))


def draw_reconstructions(ins, outs, states, shape_in, shape_state):
    """Vizualizacija ulaza i pripadajućih rekonstrkcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    plt.figure(figsize=(8, 12 * 4))
    for i in range(20):
        plt.subplot(20, 4, 4 * i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.subplot(20, 4, 4 * i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.subplot(20, 4, 4 * i + 3)
        plt.imshow(states[i][0:(shape_state[0] * shape_state[1])].reshape(shape_state),
                   vmin=-4, vmax=4, interpolation="nearest")
        plt.colorbar()
        plt.title("States")
    plt.tight_layout()


def plot_latent(inmat, labels):
    """Crtanje pozicija uzoraka u 2D latentnom prostoru
    inmat -- matrica latentnih stanja
    labels -- labela klas
    """
    plt.figure(figsize=(8, 6))
    plt.axis([-4, 4, -4, 4])
    plt.gca().set_autoscale_on(False)

    plt.scatter(inmat[:, 0], inmat[:, 1], c=np.argmax(labels, 1))
    plt.colorbar()
    plt.xlabel('z0')
    plt.ylabel('z1')


def save_latent_plot(name):
    """Spremanje trenutnog figure-a
    name -- ime datoteke
    """
    plt.savefig(name)


def weight_variable(shape, name):
    """Kreiranje težina"""
    # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    """Kreiranje pomaka"""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Prikupljanje podataka za Tensorboard"""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)


def vae_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softplus):
    """Kreiranje jednog skrivenog sloja"""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        weights = weight_variable([input_dim, output_dim], layer_name + '/weights')
        variable_summaries(weights, 'weights')
        tf.summary.tensor_summary('weightsT', weights)
        biases = bias_variable([output_dim])
        variable_summaries(biases, 'biases')
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations


tf.reset_default_graph()

sess = tf.InteractiveSession()

# definicije ulaznog tenzora
x = tf.placeholder(tf.float32, shape=[None, n_input])

# definirajte enkoiderski dio
layer_e1 = vae_layer(x, n_input, n_hidden_recog_1, 'layer_e1')
layer_e2 = vae_layer(layer_e1, n_hidden_recog_1, n_hidden_recog_2, 'layer_e2')

with tf.name_scope('z'):
    # definirajte skrivene varijable i pripadajući generator šuma
    z_mean = vae_layer(layer_e2, n_hidden_recog_2, n_z, 'z_mean', act=tf.identity)
    z_log_sigma_sq = vae_layer(layer_e2, n_hidden_recog_2, n_z, 'z_log_sigma_sq', act=tf.identity)
    eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)

    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    tf.summary.histogram('activations', z)

# definirajte dekoderski dio
layer_d1 = vae_layer(z, n_z, n_hidden_gener_1, 'layer_d1')
layer_d2 = vae_layer(layer_d1, n_hidden_gener_1, n_hidden_gener_2, 'layer_d2')

# definirajte srednju vrijednost rekonstrukcije
x_reconstr_mean = vae_layer(layer_d2, n_hidden_gener_2, n_input, 'x_reconstr_mean', act=tf.identity)

x_reconstr_mean_out = tf.nn.sigmoid(x_reconstr_mean)

# definirajte dvije komponente funkcije cijene
with tf.name_scope('cost'):
    cost1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_reconstr_mean)
    tf.summary.histogram('cross_entropy', cost1)
    cost2 = -0.5 * (1 + z_log_sigma_sq - tf.multiply(z_mean, z_mean) - tf.exp(z_log_sigma_sq))
    tf.summary.histogram('D_KL', cost2)
    cost = tf.reduce_mean(tf.reduce_sum(cost1, 1) + tf.reduce_sum(cost2, 1))  # average over batch
    tf.summary.histogram('cost', cost)

# ADAM optimizer
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Prikupljanje podataka za Tensorboard
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 100
train_writer = tf.summary.FileWriter('train', sess.graph)

sess.run(init)

total_batch = int(n_samples / batch_size)
step = 0
for epoch in range(n_epochs):
    avg_cost = 0.

    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        opt, cos = sess.run((optimizer, cost), feed_dict={x: batch_xs})
        # Compute average loss
        avg_cost += cos / n_samples * batch_size

    # Display logs per epoch step
    if epoch % (int(n_epochs / 10)) == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "cost=", "{:.9f}".format(avg_cost))
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs},
                              options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
        train_writer.add_summary(summary, epoch)

        saver.save(sess, os.path.join('train', "model.ckpt"), epoch)

train_writer.close()

# vizualizacija rekonstrukcije i stanja
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct, z_out = sess.run([x_reconstr_mean_out, z], feed_dict={x: x_sample})

if n_z == 20:
    draw_reconstructions(x_sample, x_reconstruct, z_out, (28, 28), (4, 5))  # prilagodite dimenzije prema potrebi
if n_z == 2:
    draw_reconstructions(x_sample, x_reconstruct, z_out, (28, 28), (2, 1))  # prilagodite dimenzije prema potrebi

# Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli - 1. način
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu, z_sigma = sess.run((z_mean, z_log_sigma_sq), feed_dict={x: x_sample})

plot_latent(z_mu, y_sample)
# save_latent_plot('trt.png')

# Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli - 2. način

nx = ny = 21
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28 * ny, 28 * nx))

# Trikovito popunjavanje rezultata za grid zbog fiksirane veličine z batcha u grafu
# Valjda će se to riješiti u nekoj budućoj verziji TF
Xi, Yi = np.meshgrid(x_values, y_values)
Z = np.column_stack((Xi.flatten(), Yi.flatten()))
X = np.empty((0, 28 * 28))
ind = list(range(batch_size, nx * ny, batch_size))
for i in np.array_split(Z, ind):
    if i.shape[0] < batch_size:
        i = np.concatenate((i, np.zeros((batch_size - i.shape[0], i.shape[1]))), 0)
    X = np.vstack((X, sess.run(x_reconstr_mean_out, feed_dict={z: i})))

for i, yi in enumerate(y_values):
    for j, xi in enumerate(x_values):
        canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = X[i * nx + j].reshape(28, 28)

plt.figure(figsize=(8, 10))
plt.imshow(canvas, origin="upper")
plt.xticks(np.linspace(14, 588 - 14, 11), np.round(np.linspace(-3, 3, 11), 2))
plt.yticks(np.linspace(14, 588 - 14, 11), np.round(np.linspace(3, -3, 11), 2))
plt.xlabel('z0')
plt.ylabel('z1')
plt.tight_layout()


# Vizualizacija ugašenih elemenata skrivenog sloja - 1. način

# Pomoćna funkcija za crtanje boxplot grafova
def boxplot_vis(pos, input_data, label_x, label_y):
    ax = fig.add_subplot(130 + pos)
    plt.boxplot(input_data, 0, '', 0, 0.75)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    return ax


fig = plt.figure(figsize=(15, 4))

# Vizualizacija statistike za z_mean
boxplot_vis(1, z_mu, 'Z mean values', 'Z elemets')

# Vizualizacija statistike za z_sigma
ax = boxplot_vis(2, np.square(np.exp(z_sigma)), 'Z sigma values', 'Z elemets')
ax.set_xlim([-0.05, 1.1])

# Vizualizacija statistike za težine ulaza u dekoder
test = tf.get_default_graph().get_tensor_by_name("layer_d1/weights:0")
weights_d1 = test.eval(session=sess)
boxplot_vis(3, weights_d1.T, 'Weights to decoder', 'Z elemets')

# Vizualizacija ugašenih elemenata skrivenog sloja - 2. način

from mpl_toolkits.mplot3d import Axes3D


# Funkcija za crtanje 3D bar grafa
def bargraph_vis(pos, input_data, dims, color, labels):
    ax = fig.add_subplot(120 + pos, projection='3d')
    xpos, ypos = np.meshgrid(range(dims[0]), range(dims[1]))
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = np.ones_like(zpos)
    dy = np.ones_like(zpos) * 0.5
    dz = input_data.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
    ax.view_init(elev=30., azim=5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])


fig = plt.figure(figsize=(15, 7))

# 3D bar graf za z_mean
labels = ('Samples', 'Hidden elements', 'Z mean')
bargraph_vis(1, z_mu, [5000, z_mu.shape[1]], 'g', labels)

# 3D bar graf za težine iz z_mena u dekoder
labels = ('Decoder elements', 'Hidden elements Z', 'Weights')
bargraph_vis(2, weights_d1.T, weights_d1.T.shape, 'y', labels)
plt.show()
