import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.python.keras import initializers


class GraphConvolution(Layer):
    def __init__(self, unit, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.unit = unit
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.unit])
        if self.use_bias:
            self.bias = self.add_variable("bias",
                                          shape=[self.unit])

    def call(self, inputs, **kwargs):
        if len(inputs) != 2:
            raise Exception('error')
        # featrue n * h
        feature, graph = inputs

        outputs = tf.matmul(feature, self.kernel)

        outputs = tf.matmul(graph, outputs)

        if self.use_bias:
            outputs += self.bias

        return outputs


class ShareEmbedding(Embedding):
    def __init__(self, *args, **kwargs):
        super(ShareEmbedding, self).__init__(*args, **kwargs)

    def call(self, inputs):
        output = super(ShareEmbedding, self).call(inputs)
        return output, self.embeddings


class DotSimilarity(Layer):
    def __init__(self, **kwargs):
        super(DotSimilarity, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # seq_length, batch_size, hidden_size  candidate_size, hidden_size
        query, candidate = inputs
        output = tf.einsum('bn,cn->bc', query, candidate)
        output = tf.nn.sigmoid(output)
        return output


class RelPosBiasLayer(Layer):
    def __init__(self, n_head, d_head, n_layers=None, **kwargs):
        super(RelPosBiasLayer, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_head = d_head
        self.n_layers = n_layers

    def build(self, input_shape):
        if self.n_layers:
            shape = [self.n_layers, self.n_head, self.d_head]
        else:
            shape = [self.n_head, self.d_head]

        self.r_w_bias = self.add_weight(
            'r_w_bias',
            shape=shape,
            initializer='glorot_uniform',
            dtype=self.dtype,
            trainable=True
        )
        self.r_r_bias = self.add_weight(
            'r_r_bias',
            shape=shape,
            initializer='glorot_uniform',
            dtype=self.dtype,
            trainable=True
        )
        super(RelPosBiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.r_w_bias, self.r_r_bias


def position_embedding(seq_length, units):
    pos_seq = tf.range(seq_length - 1, -1, -1.0)
    inv_freq = 1 / (10000 ** (tf.range(0, units, 2.0) / units))
    outputs = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(outputs), tf.cos(outputs)], -1)
    return pos_emb


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


class MultiHeadAtt(Layer):
    def __init__(self, units, n_head, d_head,
                 dropout, dropatt, is_training,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(MultiHeadAtt, self).__init__(**kwargs)
        self.units = units
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.is_training = is_training
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.qkv_kernel = self.add_weight(
            'qkv_kernel',
            shape=[input_shape[0][-1], 3 * self.n_head * self.d_head],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True
        )
        self.r_kernel = self.add_weight(
            'r_kernel',
            shape=[input_shape[1][-1], self.n_head * self.d_head],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True
        )
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.n_head * self.d_head, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True
        )
        super(MultiHeadAtt, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # w -> seq, bsz, hidden
        w, r, mems, r_w_bias, r_r_bias, attn_mask = inputs
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]
        scale = 1 / (self.d_head ** 0.5)
        cat = tf.concat([mems, w], axis=0) if mems is not None else w
        w_heads = tf.einsum('ibh,hd->ibd', cat, self.qkv_kernel)
        r_head_k = tf.einsum('ih,hd->id', r, self.r_kernel)

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]
        w_head_q = tf.reshape(w_head_q, [qlen, bsz, self.n_head, self.d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, self.n_head, self.d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, self.n_head, self.d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, self.n_head, self.d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, None, :, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        if self.is_training:
            attn_prob = tf.nn.dropout(attn_prob, self.dropatt)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], self.n_head * self.d_head])

        attn_out = tf.einsum('ibh,hd->ibd', attn_vec, self.kernel)
        if self.is_training:
            attn_out = tf.nn.dropout(attn_out, self.dropout)

        output = attn_out + w
        return output
