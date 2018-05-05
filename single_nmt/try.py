import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
import collections

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0

src_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/train.tok.bpe.32000.de"
tgt_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/train.tok.bpe.32000.en"
src_vocab_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/vocab.bpe.32000.de"
tgt_vocab_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/vocab.bpe.32000.en"

src_dataset = tf.data.TextLineDataset(src_file)
tgt_dataset = tf.data.TextLineDataset(tgt_file)

src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)

src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)
tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)

src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

src_tgt_dataset = src_tgt_dataset.shuffle(12800, 1234, True)

src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
                                      num_parallel_calls=4).prefetch(12800)

# Filter zero length input sequences.
src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

src_max_len = 50
tgt_max_len = 50
src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[:src_max_len], tgt[:tgt_max_len]),
                                      num_parallel_calls=4).prefetch(12800)

# Convert the word strings to ids.  Word strings that are not in the
# vocab get the lookup table's default_value integer.
src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                                      num_parallel_calls=4).prefetch(12800)
# Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src, tf.concat(([tgt_sos_id], tgt), 0), tf.concat((tgt, [tgt_eos_id]), 0)),
    num_parallel_calls=4).prefetch(12800)
# Add in sequence lengths.
src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
    num_parallel_calls=4).prefetch(12800)


# Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
def batching_func(x):
    return x.padded_batch(
        128,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused


num_buckets = 5


def key_func(u1, u2, u3, src_len, tgt_len):
    # Calculate bucket_width by maximum source sequence length.
    # Pairs with length [0, bucket_width) go to bucket 0, length
    # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
    # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
    if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
    else:
        bucket_width = 10

    # Bucket sentence pairs by the length of their source sentence and target
    # sentence.
    bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
    return tf.to_int64(tf.minimum(num_buckets, bucket_id))


def reduce_func(u1, windowed_data):
    return batching_func(windowed_data)


batched_dataset = src_tgt_dataset.apply(
    tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=128))

batched_iter = batched_dataset.make_initializable_iterator()
src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len = batched_iter.get_next()


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass


iterator = BatchedInput(
    initializer=batched_iter.initializer,
    source=src_ids,
    target_input=tgt_input_ids,
    target_output=tgt_output_ids,
    source_sequence_length=src_seq_len,
    target_sequence_length=tgt_seq_len)

initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=None)
tf.get_variable_scope().set_initializer(initializer)

with tf.variable_scope("encoder"), tf.device("/gpu:0"):
    embedding_encoder = tf.get_variable(name="embedding_encoder", shape=[500, 5], dtype=tf.float32)

with tf.variable_scope("decoder"), tf.device("/gpu:0"):
    embedding_decoder = tf.get_variable(name="embedding_decoder", shape=[500, 5], dtype=tf.float32)

batch_size = tf.size(iterator.source_sequence_length)

with tf.variable_scope("build_network"):
    with tf.variable_scope("decoder/output_projection"):
        output_layer = layers_core.Dense(500, use_bias=False, name="output_projection")

############################
# with tf.Session().as_default():
#     tf.tables_initializer().run()
#     a = src_vocab_table.lookup(tf.string_split(["not a number"]).values).eval()
# print(a)
#
# sess.run(tf.string_split(["abc\tabc", "tdf tfdd"], delimiter="[\t ]").values)

################################
with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
    with tf.variable_scope("encoder") as scope:
        encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, iterator.source)

        a1 = tf.nn.rnn_cell.DeviceWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(3), device="/gpu:0")
        a2 = tf.nn.rnn_cell.DeviceWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(3), device="/gpu:0")
        a3 = tf.nn.rnn_cell.DeviceWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(3), device="/gpu:0")
        a4 = tf.nn.rnn_cell.DeviceWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(3), device="/gpu:0")
        cell_list = [a1, a2, a3, a4]

        cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

        outputs, last_states = tf.nn.dynamic_rnn(cell, encoder_emb_inp, dtype=tf.float32,
                                                 sequence_length=iterator.source_sequence_length, swap_memory=True)
print(outputs)
print(last_states)
result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)
print(result)