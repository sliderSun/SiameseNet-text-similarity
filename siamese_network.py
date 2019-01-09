import six
import tensorflow as tf

from utils.modules import positional_encoding, multihead_attention, feedforward, gelu, swish


class SiameseNet(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, sequence_length, hidden_units):
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

        # Define rnn cells with tensorflow
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(self.config.n_layers):
                fw_cell = self.cell(hidden_units)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(self.config.n_layers):
                bw_cell = self.cell(hidden_units)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bir" + scope), tf.variable_scope("bir" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        attention_size = 128
        with tf.name_scope('attention' + scope), tf.variable_scope('attention' + scope):
            attention_w = tf.Variable(tf.truncated_normal([2 * hidden_units, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in range(sequence_length):
                u_t = self.get_activation("tanh")(tf.matmul(outputs[t], attention_w) + attention_b)
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

    @staticmethod
    def contrastive_loss(y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    @staticmethod
    def get_activation(activation_string):
        """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

        Args:
          activation_string: String name of the activation function.

        Returns:
          A Python function corresponding to the activation function. If
          `activation_string` is None, empty, or "linear", this will return None.
          If `activation_string` is not a string, it will return `activation_string`.

        Raises:
          ValueError: The `activation_string` does not correspond to a known
            activation.
        """

        # We assume that anything that"s not a string is already an activation
        # function, so we just return it.
        if not isinstance(activation_string, six.string_types):
            return activation_string

        if not activation_string:
            return None

        act = activation_string.lower()
        if act == "linear":
            return None
        elif act == "relu":
            return tf.nn.relu
        elif act == "leaky_relu":
            return tf.nn.leaky_relu
        elif act == "gelu":
            return gelu
        elif act == "swish":
            return swish
        elif act == "tanh":
            return tf.tanh
        else:
            raise ValueError("Unsupported activation: %s" % act)

    def cell(self, n_hidden):
        if self.config.cell == "sru":
            fw_cell = tf.contrib.rnn.SRUCell(n_hidden)
        elif self.config.cell == "gru":
            fw_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        elif self.config.cell == "indyLSTM":
            fw_cell = tf.contrib.rnn.IndyLSTMCell(n_hidden)
        else:
            fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=0.9, state_is_tuple=True)
        return fw_cell

    def cnn_layer(self, inputs, scope):
        filters = [2, 3, 4, 5]
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs = []
        channel = self.config.num_units // len(filters)
        for ii, width in enumerate(filters):
            with tf.variable_scope(scope + "_cnn_{}_layer".format(ii)):
                weight = tf.Variable(
                    tf.truncated_normal([width, self.config.num_units, 1, channel], stddev=0.1, name='w'))
                bias = tf.get_variable('bias', [channel], initializer=tf.constant_initializer(0.0))
                output = tf.nn.conv2d(inputs, weight, strides=[1, 1, self.config.num_units, 1], padding='SAME')
                output = tf.nn.bias_add(output, bias, data_format="NHWC")
                output = self.get_activation("gelu")(output)
                output = tf.reshape(output, shape=[-1, self.config.max_document_length, channel])
                outputs.append(output)
        outputs = tf.concat(outputs, axis=-1)
        return outputs

    def transformer(self, embed, value, scope):
        with tf.variable_scope(scope + "_Transformer_Encoder"):
            # Positional Encoding
            embed += positional_encoding(value, num_units=self.config.num_units, zero_pad=False, scale=False,
                                         scope=scope + "_post")
            # Dropout
            output = self.multi_head_block(embed, scope)
            return output

    def multi_head_block(self, query, scope, causality=False):
        """
        多头注意力机制
        :param scope:
        :param query:
        :param causality:
        :return:
        """
        for i in range(self.config.num_blocks):
            with tf.variable_scope(scope + "_num_blocks_{}".format(i)):
                # multi head Attention ( self-attention)
                query = multihead_attention(
                    queries=query, keys=query, num_units=self.config.num_units, num_heads=self.config.num_heads,
                    dropout_rate=self.config.dropout_keep_prob, is_training=True, causality=causality,
                    scope=scope + "self_attention")
                # Feed Forward
                query = feedforward(query, num_units=[4 * self.config.num_units, self.config.num_units])
        return query

    def __init__(self, config, vocab_size):
        # Placeholders for input, output and dropout
        self.config = config
        self.input_x1 = tf.placeholder(tf.int32, [None, self.config.max_document_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, self.config.max_document_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.initializer = None
        if self.config.initializer == "normal":
            self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        elif self.config.initializer == "glorot":
            self.initializer = tf.glorot_uniform_initializer()
        elif self.config.initializer == "xavier":
            self.initializer = tf.glorot_normal_initializer()
        else:
            raise ValueError("Unknown initializer")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.get_variable('lookup_table', dtype=tf.float32,
                                     shape=[vocab_size, self.config.embedding_dim],
                                     initializer=self.initializer,
                                     trainable=True)
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        with tf.name_scope("output"):
            # add transformer layer
            # output1 = self.transformer(self.embedded_chars1, self.input_x1, "side1")
            # output2 = self.transformer(self.embedded_chars2, self.input_x2, "side2")
            # add cnn layer
            output1 = self.cnn_layer(self.embedded_chars1, "side1")
            output2 = self.cnn_layer(self.embedded_chars2, "side2")

            self.out1 = self.BiRNN(output1, self.dropout_keep_prob, "side1", self.config.max_document_length,
                                   self.config.hidden_units)
            self.out2 = self.BiRNN(output2, self.dropout_keep_prob, "side2", self.config.max_document_length,
                                   self.config.hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keepdims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, self.config.batch_size)
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.round(self.distance),
                                        name="temp_sim")  # auto threshold 0.4
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.r, _ = tf.metrics.recall(self.input_y, self.temp_sim, name='recall')
            self.f1score = (2 * self.accuracy * self.r) / (self.accuracy + self.r)
            self.confusion_matrix = tf.confusion_matrix(self.input_y, self.temp_sim)
