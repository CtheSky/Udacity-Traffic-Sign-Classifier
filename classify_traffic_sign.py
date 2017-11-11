import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import dropout
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import max_pool2d

# Model
X = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

reg_constant = 1e-5
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

with tf.name_scope('model'):
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = conv2d(inputs=X,
                   num_outputs=6,
                   kernel_size=(5, 5),
                   stride=(1, 1),
                   padding='valid',
                   weights_regularizer=l2_regularizer(scale=reg_constant),
                   normalizer_fn=batch_norm,
                   normalizer_params=bn_params,
                   activation_fn=tf.nn.relu,
                   scope='conv1')

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = max_pool2d(conv1, kernel_size=(2, 2))

    # Convolutional. Output = 10x10x16.
    conv2 = conv2d(inputs=conv1,
                   num_outputs=16,
                   kernel_size=(5, 5),
                   stride=(1, 1),
                   padding='valid',
                   weights_regularizer=l2_regularizer(scale=reg_constant),
                   normalizer_fn=batch_norm,
                   normalizer_params=bn_params,
                   activation_fn=tf.nn.relu,
                   scope='conv2')

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = max_pool2d(conv2, kernel_size=(2, 2))

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 1024.
    fc1 = fully_connected(fc0, 1024,
                          activation_fn=tf.nn.elu,
                          weights_regularizer=l2_regularizer(scale=reg_constant),
                          normalizer_fn=batch_norm,
                          normalizer_params=bn_params,
                          scope='fc1')
    fc1 = dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 1024. Output = 1024.
    fc2 = fully_connected(fc1, 1024,
                          activation_fn=tf.nn.elu,
                          weights_regularizer=l2_regularizer(scale=reg_constant),
                          normalizer_fn=batch_norm,
                          normalizer_params=bn_params,
                          scope='fc2')
    fc2 = dropout(fc1, keep_prob)

    # Layer 5: Fully Connected. Input = 1024. Output = 43.
    logits = fully_connected(fc2, 43,
                             activation_fn=None,
                             weights_regularizer=l2_regularizer(scale=reg_constant),
                             normalizer_fn=batch_norm,
                             normalizer_params=bn_params,
                             scope='fc3')

# command line flags
flags = tf.app.flags
flags.DEFINE_string('filename', '', "cropped traffic sign image")


def main(_):

    # load traffic sign names
    import csv
    with open('signnames.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        id2name = {int(row['ClassId']): row['SignName'] for row in reader}

    # load image
    img_content = tf.read_file(flags.FLAGS.filename)
    img = tf.image.decode_jpeg(img_content, channels=3)
    img = tf.image.resize_images(img, (32, 32))
    img = tf.image.rgb_to_grayscale(img)

    # restore and run the model
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './tmp/model.ckpt')

        values, cids = sess.run(tf.nn.top_k(tf.nn.softmax(logits), 5), feed_dict={is_training: False, keep_prob: 1.0, X: [img.eval()]})
        for i in range(5):
            val, cid = values[0][i], cids[0][i]
            print('{:2f} confidence for {}'.format(val, id2name[cid]))


if __name__ == '__main__':
    tf.app.run()
