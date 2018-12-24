import tensorflow as tf


class SiameseNet(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden = hidden_units
        n_layers = 3
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.9, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.9, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bir" + scope), tf.variable_scope("bir" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        attention_size = 128
        with tf.name_scope('attention' + scope), tf.variable_scope('attention' + scope):
            attention_w = tf.Variable(tf.truncated_normal([2 * hidden_units, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in range(sequence_length):
                u_t = tf.nn.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1]), name='attention_uw')
            attn_z = []
            for t in range(sequence_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [sequence_length, -1, 1])
            return tf.reduce_sum(outputs * alpha_trans, 0)

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def contro_loss(self, distance, input_y):
        '''
        总结下来对比损失的特点：首先看标签，然后标签为1是正对，负对部分损失为0，最小化总损失就是最小化类内损失(within_loss)部分，
        让s逼近margin的过程，是个增大的过程；标签为0是负对，正对部分损失为0，最小化总损失就是最小化between_loss，而且此时between_loss就是s，
        所以这个过程也是最小化s的过程，也就使不相似的对更不相似了
        '''
        one = tf.constant(1.0)
        margin = 1.0
        y_true = tf.to_float(input_y)

        # 类内损失：
        max_part = tf.square(tf.maximum(margin - distance, 0))  # margin是一个正对该有的相似度临界值
        within_loss = tf.multiply(y_true, max_part)  # 如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s

        # 类间损失：
        between_loss = tf.multiply(one - y_true,
                                   distance)  # 如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似

        # 总体损失（要最小化）：
        loss = 0.5 * tf.reduce_mean(within_loss + between_loss)
        return loss

    def __init__(
            self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                # tf.truncated_normal([vocab_size, embedding_size], stddev=0.1),
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True, name="W")
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            # self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            # self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)

        # Create a convolution + maxpool layer for each filter size

        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size,
                                   sequence_length, hidden_units)
            self.out2 = self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size,
                                   sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keepdims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
            # self.loss = self.contro_loss(self.distance, self.input_y)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.round(self.distance),
                                        name="temp_sim")  # auto threshold 0.4
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
