import tensorflow as tf
from layers import GraphConvolution, MultiHeadAtt, position_embedding, RelPosBiasLayer, DotSimilarity
from tensorflow.keras import layers as ly


class GCN(tf.keras.Model):
    def __init__(self, units, vocab_size, n_layer=2):
        super(GCN, self).__init__()
        self.gcn_layers = []
        for i in range(n_layer):
            layer = GraphConvolution(units)
            self.gcn_layers.append(layer)
        self.last_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None, mask=None):
        feature, graph = inputs
        outputs = feature
        for layer in self.gcn_layers:
            outputs = layer([outputs, graph])
        outputs = self.last_layer(outputs)
        return outputs


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

    return tf.stop_gradient(new_mem)


# def _create_mask(qlen, mlen, mems_mask, same_length=False):
#     attn_mask = tf.ones([qlen, qlen])
#     mask_u = tf.matrix_band_part(attn_mask, 0, -1)
#     mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
#     attn_mask_pad = tf.zeros([qlen, mlen])
#     ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
#     if same_length:
#         mask_l = tf.matrix_band_part(attn_mask, -1, 0)
#         ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
#     return ret


def _create_mask(dec_inp, mlen, s_index):
    # qlen, batch_size
    inp_mask = tf.cast(dec_inp != 0, tf.int32)
    if mlen == 0:
        # mlen, batch_size
        mems_mask = tf.tile(tf.cast(dec_inp[0, :] != s_index, tf.int32), [mlen, 1])
        mask = tf.concat([mems_mask, inp_mask], axis=0)
    else:
        mask = inp_mask
    return mask


class TransformerXL(tf.keras.Model):
    def __init__(self, embedding_size, embedding_dim,
                 d_model, n_head, d_head, d_inner,
                 dropout, dropatt, is_training,
                 n_layers, mem_len,
                 s_index, e_index,
                 rel_t_r, rel_r_f,
                 untie_r=False,
                 **kwargs):
        super(TransformerXL, self).__init__(**kwargs)
        self.d_model = d_model
        self.dropout = dropout
        self.n_layers = n_layers
        self.mem_len = mem_len
        self.s_index = s_index
        self.e_index = e_index
        self.untie_r = untie_r
        self.rel_t_r = tf.constant(rel_t_r)
        self.rel_r_f = tf.constant(rel_r_f)
        if untie_r:
            self.rel_pos_bias_layer = RelPosBiasLayer(n_head, d_head, n_layers)
        else:
            self.rel_pos_bias_layer = RelPosBiasLayer(n_head, d_head)
        theme_size, role_size, form_size, word_size = embedding_size
        theme_dim, role_dim, form_dim, word_dim = embedding_dim
        self.theme_size = theme_size
        self.role_size = role_size
        self.form_size = form_size
        self.embedding_layers = dict()
        self.embedding_layers['theme'] = ly.Embedding(theme_size, theme_dim)
        self.embedding_layers['role'] = ly.Embedding(role_size, role_dim)
        self.embedding_layers['form'] = ly.Embedding(form_size, form_dim)
        self.embedding_layers['word'] = ly.Embedding(word_size, word_dim, mask_zero=True)
        self.theme_layer = ly.Dense(d_model, activation='tanh')
        self.blocks = []
        for i in range(n_layers):
            block = []
            att = MultiHeadAtt(units=d_model,
                               n_head=n_head,
                               d_head=d_head,
                               dropout=dropout,
                               dropatt=dropatt,
                               is_training=is_training)
            block.append(att)
            block.append(ly.LayerNormalization(axis=-1))
            block.append(ly.Dense(d_inner, activation='relu'))
            block.append(ly.Dense(d_model))
            block.append(ly.LayerNormalization(axis=-1))
            self.blocks.append(block)
        self.output_layer = DotSimilarity()

    def call(self, inputs, training=None, mask=None):
        dec_inp, mems = inputs

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen
        new_mems = []
        r_w_bias, r_r_bias = self.rel_pos_bias_layer(None)

        # theme_size, hidden_size
        theme_embedding = self.embedding_layers['theme'](tf.range(0, self.theme_size))
        # role_size, hidden_size
        role_embedding = self.embedding_layers['role'](tf.range(0, self.role_size))
        # form_size, hidden_size
        form_embedding = self.embedding_layers['form'](tf.range(0, self.form_size))

        f_r_embedding = tf.einsum('rf,fh->rh', self.rel_r_f, form_embedding)
        role_embedding = tf.concat([role_embedding, f_r_embedding], axis=-1)
        r_t_embedding = tf.einsum('tr,rh->th', self.rel_t_r, role_embedding)
        theme_embedding = tf.concat([theme_embedding, r_t_embedding], axis=-1)
        theme_embedding = self.theme_layer(theme_embedding)

        output = self.embedding_layers['word'](dec_inp)
        # attn_mask = _create_mask(qlen, mlen, False)
        key_mask = _create_mask(dec_inp, mlen, self.s_index)
        pos_emb = position_embedding(klen, self.d_model)

        if training:
            output = tf.nn.dropout(output, self.dropout)
            pos_emb = tf.nn.dropout(pos_emb, self.dropout)

        if mems is None:
            mems = [None] * self.n_layers
        for i in range(self.n_layers):
            new_mems.append(_cache_mem(output, mems[i], self.mem_len))
            block = self.blocks[i]
            _rw_bias = r_w_bias[i] if self.untie_r else r_w_bias
            _rr_bias = r_r_bias[i] if self.untie_r else r_r_bias
            output = block[0]([output, pos_emb, mems[i], _rw_bias, _rr_bias, key_mask])
            _temp = output
            _temp = block[1](_temp)
            _temp = block[2](_temp)
            if training:
                _temp = tf.nn.dropout(_temp, self.dropout)
            _temp = block[3](_temp)
            if training:
                _temp = tf.nn.dropout(_temp, self.dropout)
            output = block[4](output + _temp)

        if training:
            output = tf.nn.dropout(output, self.dropout)

        # seq_length, batch_size
        mask = tf.cast(dec_inp == self.e_index, dtype=tf.int32)
        mask = mask[:, :, None]
        output = tf.reduce_sum(mask * output, axis=0)
        output = self.output_layer([output, theme_embedding])
        return output





