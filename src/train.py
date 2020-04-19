import tensorflow as tf
import argparse
import numpy as np
from model import GCN, TransformerXL
from utils import SPOClassifyData, print_metric

# tensorflow 2.0 train pattern
# 1.data input
# 2.custom layer
# class layer(tf.keras.layers.Layer):
#     def __init__(self):
#         parameter
#     def build(self, input_shape):
#         define variables
#     def call(self, inputs, **kwargs):
#         compute logic
# 3.model from merge layer
# class Model(tf.keras.Model):
#     def __init__(self):
#         define layer
#     def call(self, inputs, training=None, mask=None):
#         layer compute logic
# def model_func(inputs):
#     layer define
#     layer compute logic
# model = Model()
# model = tf.keras.Model(inputs, model_func(inputs))
# 4.define loss and metric records
# metric records
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# 5.define optimizer and learning rate drop strategy
# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, **kwargs)
# 6.save and restore
# ckpt = tf.train.Checkpoint()
# ckpt_manager = tf.train.CheckpointManager(ckpt, save_path, **kwargs)
# ckpt.restore()
# ckpt_manager.save()
# 7.train step function
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         outputs = model(inputs, **kwargs)
#         loss = loss_func(targets, outputs)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     train_loss(loss)
#     train_metric(targets, outputs)
# 8.多GPU
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
# 2个特殊的地方
# 一个是输入的分发器
# dist_dataset = strategy.experimental_distribute_dataset(dataset)
# 一个是输出的合成器
# @tf.function
# def distributed_train_step(dataset_inputs, dataset_labels):
#     per_replica_losses = strategy.experimental_run_v2(
#         train_step, args=(dataset_inputs, dataset_labels)
#     )
#     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
_EPSILON = 1e-7


def sigmoid_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    _y = tf.clip_by_value(y_pred, _EPSILON, 1. - _EPSILON)
    loss = -1.0 * tf.math.reduce_mean(y_true * tf.math.log(_y) + (1 - y_true) * tf.math.log(1 - _y))
    return loss


@tf.function
def train_step(inputs, targets, model, optimizer):
    with tf.GradientTape() as tape:
        outputs, mems = model(inputs)
        loss = sigmoid_loss(targets, outputs)
    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss, mems


@tf.function
def test_step(inputs, targets, model):
    outputs = model(inputs)
    loss = sigmoid_loss(targets, outputs)
    return loss


def train(args):
    data_loader = SPOClassifyData(args.data_path, './data')
    model = TransformerXL(
        embedding_size=[len(data_loader.schema[0]), len(data_loader.schema[1]), len(data_loader.schema[2]), len(data_loader.vocab)],
        embedding_dim=[int(args.embedding_dim / 2), int(args.embedding_dim / 2), int(args.embedding_dim / 2), args.embedding_dim],
        d_model=args.d_model,
        n_head=args.n_head,
        d_head=args.d_head,
        d_inner=args.d_inner,
        dropout=args.dropout,
        dropatt=args.dropatt,
        is_training=True,
        n_layers=args.n_layers,
        mem_len=args.mem_length,
        s_index=data_loader.vocab['<S>'],
        e_index=data_loader.vocab['<E>'],
        rel_t_r_f=data_loader.schema[3],
        untie_r=args.untie_r
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate,
        decay_steps=2000,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if args.load_path:
        model.load_weights(args.load_model)

    i = 0
    while i < args.epoch_num:
        train_generator, test_generator = data_loader.get_batch_generator(args.batch_size, args.seq_length)
        mems = [np.zeros([args.mem_length, args.batch_size, args.d_model], dtype=np.float32) for layer in range(args.n_layers)]
        for step, batch_data in enumerate(train_generator):
            x, y = batch_data
            loss, mems = train_step([x, mems], y, model, optimizer)
            if step % 100 == 0:
                print_metric({'loss': loss}, style='train')
        model.save(args.model_path)

        loss = 0.0
        for batch_data in test_generator:
            x, y = batch_data
            loss += test_step(x, y, model)
        print_metric({'loss': loss}, style='eval')

        i += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--mem_length', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dropatt', type=float, default=0.0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_head', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_inner', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--untie_r', action='store_true', default=False)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
