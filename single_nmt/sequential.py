import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import lookup_ops
import argparse
from nmt.scripts import bleu, rouge
import numpy as np
import os
import time
import codecs
import collections
import re
import random
import math
import abc
import six
from single_nmt.config import prefix

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


# =======================================================================
# hparams
# =======================================================================

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--encoder_type", type=str, default="uni", help="""\
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.\
      """)
    parser.add_argument("--residual", type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to add residual connections.")
    parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                        default=True,
                        help="Whether to use time-major mode for dynamic RNN.")
    parser.add_argument("--num_embeddings_partitions", type=int, default=0,
                        help="Number of partitions for embedding vars.")

    # attention mechanisms
    parser.add_argument("--attention", type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
    parser.add_argument(
        "--attention_architecture",
        type=str,
        default="standard",
        help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
    parser.add_argument(
        "--output_attention", type="bool", nargs="?", const=True,
        default=True,
        help="""\
      Only used in standard attention_architecture. Whether use attention as
      the cell output at each timestep.
      .\
      """)
    parser.add_argument(
        "--pass_hidden_state", type="bool", nargs="?", const=True,
        default=True,
        help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--warmup_scheme", type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
    parser.add_argument(
        "--decay_scheme", type=str, default="", help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))

    # initializer
    parser.add_argument("--init_op", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))

    # data
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix, e.g., en.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="Target suffix, e.g., de.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    # Vocab
    parser.add_argument("--vocab_prefix", type=str, default=None, help="""\
      Vocab prefix, expect files with src/tgt suffixes.\
      """)
    parser.add_argument("--embed_prefix", type=str, default=None, help="""\
      Pretrained embedding prefix, expect files with src/tgt suffixes.
      The embedding files should be Glove formated txt files.\
      """)
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True,
                        default=False,
                        help="""\
      Whether to use the source vocab and embeddings for both source and
      target.\
      """)
    parser.add_argument("--check_special_token", type="bool", default=True,
                        help="""\
                      Whether check special sos, eos, unk tokens exist in the
                      vocab files.\
                      """)

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""\
      Max length of tgt sequences during inference.  Also use to restrict the
      maximum decoding length.\
      """)

    # Default settings works well (rarely need to change)
    parser.add_argument("--unit_type", type=str, default="lstm",
                        help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")

    # SPM
    parser.add_argument("--subword_option", type=str, default="",
                        choices=["", "bpe", "spm"],
                        help="""\
                      Set to bpe or spm to activate subword desegmentation.\
                      """)

    # Misc
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of gpus in each worker.")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--metrics", type=str, default="bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,rouge,accuracy)"))
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
      How many training steps to do per external evaluation.  Automatically set
      based on data if None.\
      """)
    parser.add_argument("--scope", type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from FLAGS."))
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--override_loaded_hparams", type="bool", nargs="?",
                        const=True, default=False,
                        help="Override loaded hparams with values specified")
    parser.add_argument("--num_keep_ckpts", type=int, default=5,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--avg_ckpts", type="bool", nargs="?",
                        const=True, default=False, help=("""\
                      Average the last N checkpoints for external evaluation.
                      N can be controlled by setting --num_keep_ckpts.\
                      """))

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--inference_list", type=str, default=None,
                        help=("A comma-separated list of sentence indices "
                              "(0-based) to decode."))
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
      Reference file to compute evaluation scores (if provided).\
      """))
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--sampling_temperature", type=float,
                        default=0.0,
                        help=("""\
      Softmax sampling temperature for inference decoding, 0.0 means greedy
      decoding. This option is ignored when using beam search.\
      """))
    parser.add_argument("--num_translations_per_input", type=int, default=1,
                        help=("""\
      Number of translations generated for each sentence. This is only used for
      inference.\
      """))

    # Job info
    parser.add_argument("--jobid", type=int, default=0,
                        help="Task id of the worker.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers (inference only).")
    parser.add_argument("--num_inter_threads", type=int, default=0,
                        help="number of inter_op_parallelism_threads")
    parser.add_argument("--num_intra_threads", type=int, default=0,
                        help="number of intra_op_parallelism_threads")


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        out_dir=flags.out_dir,

        # Networks
        num_units=flags.num_units,
        num_layers=flags.num_layers,  # Compatible
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_partitions=flags.num_embeddings_partitions,

        # Attention mechanisms
        attention=flags.attention,
        attention_architecture=flags.attention_architecture,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,

        # Vocab
        sos=flags.sos if flags.sos else SOS,
        eos=flags.eos if flags.eos else EOS,
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,

        # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(","),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
    )


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None, eos=None, unk=None):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    if tf.gfile.Exists(vocab_file):
        print("# Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            # Verify if the vocab starts with unk, sos, eos
            # If not, prepend those tokens & generate a new vocab file
            if not unk: unk = UNK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                print("The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                        tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file '%s' does not exist." % vocab_file)

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def extend_hparams(hparams):
    """Extend training hparams."""
    assert hparams.num_encoder_layers and hparams.num_decoder_layers
    # 不同的encoder和decoder layers, encoder和decoder不共享hidden_state
    if hparams.num_encoder_layers != hparams.num_decoder_layers:
        hparams.pass_hidden_state = False
        print("Num encoder layer %d is different from num decoder layer"
              " %d, so set pass_hidden_state to False" % (
                  hparams.num_encoder_layers,
                  hparams.num_decoder_layers))

    # Sanity checks，校验双向lstm和gnmt
    if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
        raise ValueError("For bi, num_encoder_layers %d should be even" %
                         hparams.num_encoder_layers)
    if (hparams.attention_architecture in ["gnmt"] and
            hparams.num_encoder_layers < 2):
        raise ValueError("For gnmt attention architecture, "
                         "num_encoder_layers %d should be >= 2" %
                         hparams.num_encoder_layers)

    # Set residual layers，残差连接
    # Google’s Neural Machine Translation System
    # https://arxiv.org/pdf/1609.08144.pdf
    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if hparams.residual:
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1

        if hparams.encoder_type == "gnmt":
            # The first unidirectional layer (after the bi-directional layer) in
            # the GNMT encoder can't have residual connection due to the input is
            # the concatenation of fw_cell and bw_cell's outputs.
            num_encoder_residual_layers = hparams.num_encoder_layers - 2

            # Compatible for GNMT models
            if hparams.num_encoder_layers == hparams.num_decoder_layers:
                num_decoder_residual_layers = num_encoder_residual_layers
    hparams.add_hparam("num_encoder_residual_layers", num_encoder_residual_layers)
    hparams.add_hparam("num_decoder_residual_layers", num_decoder_residual_layers)

    # SentencePiece Model: https://github.com/google/sentencepiece
    # Byte Pair Encoding:  https://arxiv.org/pdf/1508.07909.pdf
    if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
        raise ValueError("subword option must be either spm, or bpe")

    # Flags
    print("# hparams:")
    print("  src=%s" % hparams.src)
    print("  tgt=%s" % hparams.tgt)
    print("  train_prefix=%s" % hparams.train_prefix)
    print("  dev_prefix=%s" % hparams.dev_prefix)
    print("  test_prefix=%s" % hparams.test_prefix)
    print("  out_dir=%s" % hparams.out_dir)

    # Vocab
    # Get vocab file names first
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    # Source vocab, 添加UNK, SOS, EOS
    src_vocab_size, src_vocab_file = check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=UNK)

    # Target vocab, 添加UNK, SOS, EOS
    if hparams.share_vocab:
        print("  using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=hparams.check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

    # Pretrained Embeddings:
    hparams.add_hparam("src_embed_file", "")
    hparams.add_hparam("tgt_embed_file", "")
    if hparams.embed_prefix:
        src_embed_file = hparams.embed_prefix + "." + hparams.src
        tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

        if tf.gfile.Exists(src_embed_file):
            hparams.src_embed_file = src_embed_file

        if tf.gfile.Exists(tgt_embed_file):
            hparams.tgt_embed_file = tgt_embed_file

    # Check out_dir
    if not tf.gfile.Exists(hparams.out_dir):
        print("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)

    # Evaluation, bleu, rouge
    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
            best_metric_dir = os.path.join(hparams.out_dir, "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams


# make hparams ===================================================================
my_args = f"""nmt.nmt \
--src=de --tgt=en \
--vocab_prefix={prefix}/wmt16/wmt16_de_en/vocab.bpe.32000  \
--train_prefix={prefix}/wmt16/wmt16_de_en/train.tok.bpe.32000 \
--dev_prefix={prefix}/wmt16/wmt16_de_en/newstest2016.tok.bpe.32000  \
--test_prefix={prefix}/wmt16/wmt16_de_en/newstest2015.tok.bpe.32000 \
--out_dir={prefix}/wmt16/wmt16-model-single \
--attention=luong \
--batch_size=2 \
--num_train_steps=4000 \
--steps_per_stats=100 \
--num_layers=3 \
--num_units=4 \
--dropout=0.2 \
--metrics=bleu
""".replace("=", " ").split()

nmt_parser = argparse.ArgumentParser()  # 创建args parser
add_arguments(nmt_parser)  # 添加需要的参数选项
flags, unparsed = nmt_parser.parse_known_args(my_args)  # 传递参数，排除冲突

default_hparams = create_hparams(flags)  # 转为tf的hparams
hparams = extend_hparams(default_hparams)

src_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/train.tok.bpe.32000.de"
tgt_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/train.tok.bpe.32000.en"
src_vocab_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/vocab.bpe.32000.de"
tgt_vocab_file = "/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/vocab.bpe.32000.en"

num_buckets = 5

time_major = hparams.time_major

num_encoder_layers = hparams.num_encoder_layers
num_decoder_layers = hparams.num_decoder_layers

num_encoder_residual_layers = hparams.num_encoder_residual_layers
num_decoder_residual_layers = hparams.num_decoder_residual_layers

mode = tf.contrib.learn.ModeKeys.TRAIN

# # create_train_model ----------------------

# iterator -----------------------------------
src_dataset = tf.data.TextLineDataset(src_file)
tgt_dataset = tf.data.TextLineDataset(tgt_file)
src_vocab_table = lookup_ops.index_table_from_file(src_vocab_file, default_value=UNK_ID)
tgt_vocab_table = lookup_ops.index_table_from_file(tgt_vocab_file, default_value=UNK_ID)
src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(EOS)), tf.int32)
tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(SOS)), tf.int32)
tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(EOS)), tf.int32)
src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
src_tgt_dataset = src_tgt_dataset.shuffle(hparams.batch_size * 100, 1234, True)
src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
    num_parallel_calls=4).prefetch(hparams.batch_size * 100)
# Filter zero length input sequences.
src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
src_max_len = 50
tgt_max_len = 50
src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (src[:src_max_len], tgt[:tgt_max_len]),
                                      num_parallel_calls=4).prefetch(hparams.batch_size * 100)
# Convert the word strings to ids.  Word strings that are not in the
# vocab get the lookup table's default_value integer.
src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
                                      num_parallel_calls=4).prefetch(hparams.batch_size * 100)
# Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt: (src, tf.concat(([tgt_sos_id], tgt), 0), tf.concat((tgt, [tgt_eos_id]), 0)),
    num_parallel_calls=4).prefetch(hparams.batch_size * 100)
# Add in sequence lengths.
src_tgt_dataset = src_tgt_dataset.map(
    lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
    num_parallel_calls=4).prefetch(hparams.batch_size * 100)


# Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
def batching_func(x):
    return x.padded_batch(
        hparams.batch_size,
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
    tf.contrib.data.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=hparams.batch_size))
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

batch_size = tf.size(iterator.source_sequence_length)
# ------------------------------------------


# Initializer ---------------------------------
initializer = tf.random_uniform_initializer(-hparams.init_weight, hparams.init_weight, seed=None)
tf.get_variable_scope().set_initializer(initializer)
# -------------------------------------------


# Embeddings -----------------------------------
src_embed_size = hparams.num_units
with tf.variable_scope("encoder"), tf.device("/gpu:0"):
    embedding_encoder = tf.get_variable(name="embedding_encoder",
                                        shape=[hparams.src_vocab_size, src_embed_size],
                                        dtype=tf.float32)
tgt_embed_size = hparams.num_units
with tf.variable_scope("decoder"), tf.device("/gpu:0"):
    embedding_decoder = tf.get_variable(name="embedding_decoder",
                                        shape=[hparams.tgt_vocab_size, tgt_embed_size],
                                        dtype=tf.float32)
# ----------------------------------------------


# Projection ------------------------------------
with tf.variable_scope("build_network"):
    with tf.variable_scope("decoder/output_projection"):
        output_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")


# ------------------------------------------------


# # Test ----------------------------------------
# config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
# config_proto.gpu_options.allow_growth = True
# sess = tf.Session(config=config_proto)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.tables_initializer())
# sess.run(iterator.initializer)
# for i in range(3):
#     once = sess.run(iterator.source)
#     print(once)
#     print(once.shape)
# sess.close()
# print(hparams.tgt_vocab_size)
# # ---------------------------------------------


def _single_cell(unit_type, num_units, forget_bias, dropout, mode):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    # Cell Type
    if unit_type == "lstm":
        print("  LSTM, forget_bias=%g" % forget_bias, end="")
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        print("  GRU", end="")
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        print("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
              end="")
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    elif unit_type == "nas":
        print("  NASCell", end="")
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        print("  %s, dropout=%g " % (type(single_cell).__name__, dropout))

    # Device Wrapper
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, "/gpu:0")

    return single_cell


def create_rnn_cell(unit_type, num_units, num_layers, forget_bias, dropout, mode):
    """Create multi-layer RNN cell.

    Args:
      unit_type: string representing the unit type, i.e. "lstm".
      num_units: the depth of each unit.
      num_layers: number of cells.
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
        cells in the returned list will be wrapped with `ResidualWrapper`.
      forget_bias: the initial forget bias of the RNNCell(s).
      dropout: floating point value between 0.0 and 1.0:
        the probability of dropout.  this is ignored if `mode != TRAIN`.
      mode: either tf.contrib.learn.TRAIN/EVAL/INFER
      num_gpus: The number of gpus to use when performing round-robin
        placement of layers.
      base_gpu: The gpu device id to use for the first RNN cell in the
        returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
        as its device id.
      single_cell_fn: allow for adding customized cell.
        When not specified, we default to model_helper._single_cell
    Returns:
      An `RNNCell` instance.
    """
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode
        )
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


# Encoder ----------------------------------------------------------
"""Build an encoder."""
with tf.variable_scope("encoder") as encoder_scope:
    # Look up embedding,
    # iterator.source:                [batch_size, max_time]
    # tf.transpose(iterator.source):  [max_time, batch_size]
    # encoder_emb_inp:                [max_time, batch_size, num_units]
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, tf.transpose(iterator.source))
    # MultiRNNCell, LSTM(num_units) * num_encoder_layers
    encode_cell = create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=hparams.num_encoder_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=mode)
    # Encoder_outputs: time_major=True时，
    # encoder_emb_inp:[max_time, batch_size, num_units]
    # encoder_outputs:[max_time, batch_size, encode_cell.output_size]  encode_cell.output_size = num_units
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        encode_cell,
        encoder_emb_inp,
        dtype=encoder_scope.dtype,
        sequence_length=iterator.source_sequence_length,
        time_major=True,
        swap_memory=True)
    # # Test ----------------------------------------
    # config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config_proto.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_proto)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.tables_initializer())
    # sess.run(iterator.initializer)
    # for i in range(3):
    #     p = sess.run([encoder_outputs, encoder_state, iterator.source_sequence_length])
    #     print(p[0])
    #     print(p[0].shape)
    #     for x in p[1]:
    #         for x0 in x:
    #             print(x0.shape)
    #     print(p[2])
    #     print("----------------------")
    # sess.close()
    # # ---------------------------------------------


# -----------------------------------------------------------


def create_attention_mechanism(attention_option, num_units, memory, source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # Mechanism
    if attention_option == "luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


# Decoder -------------------------------------------------
"""Build and run a RNN decoder with a final projection layer.

Args:
  encoder_outputs: The outputs of encoder for every time step.
  encoder_state: The final state of the encoder.
  hparams: The Hyperparameters configurations.

Returns:
  A tuple of final logits and final decoder state:
    logits: size [time, batch_size, vocab_size] when time_major=True.
"""
with tf.variable_scope("decoder") as decoder_scope:
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    # Ensure memory is batch-major: memory: [batch_size, max_time, num_units]
    memory = tf.transpose(encoder_outputs, [1, 0, 2])
    attention_mechanism = create_attention_mechanism(
        hparams.attention, hparams.num_units, memory, iterator.source_sequence_length)
    decoder_cell = create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=mode)
    decoder_cell = seq2seq.AttentionWrapper(
        decoder_cell,
        attention_mechanism,
        attention_layer_size=hparams.num_units,
        alignment_history=False,
        output_attention=hparams.output_attention,
        name="attention")
    # TODO(thangluong): do we need num_layers, num_gpus?
    decoder_cell = tf.contrib.rnn.DeviceWrapper(decoder_cell, "/gpu:0")
    # IF hparams.pass_hidden_state:
    decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
    # decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32)
    # decoder_emp_inp:                    [max_time, batch_size, num_units]
    # tf.transpose(iterator.target_input):[max_time, batch_size, num_units]
    # 输入为target_input
    decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, tf.transpose(iterator.target_input))
    # Helper import tensorflow.contrib.seq2seq.python.ops.helper
    helper = seq2seq.TrainingHelper(decoder_emb_inp, iterator.target_sequence_length, time_major=True)
    # Decoder import tensorflow.contrib.seq2seq.python.ops.basic_decoder
    my_decoder = seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state)
    # Dynamic decoding import tensorflow.contrib.seq2seq.python.ops.decoder
    outputs, final_context_state, _ = seq2seq.dynamic_decode(
        my_decoder,
        output_time_major=True,
        swap_memory=True,
        scope=decoder_scope)
    sample_id = outputs.sample_id
    # Note: there's a subtle difference here between train and inference.
    # We could have set output_layer when create my_decoder
    #   and shared more code between train and inference.
    # We chose to apply the output_layer to all timesteps for speed:
    #   10% improvements for small models & 20% for larger ones.
    # If memory is a concern, we should apply output_layer per timestep.
    logits = output_layer(outputs.rnn_output)  # projection

    # # Test ----------------------------------------
    # config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config_proto.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_proto)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.tables_initializer())
    # sess.run(iterator.initializer)
    # for i in range(3):
    #     p_output, p_context, src_len, tgt_len = sess.run(
    #         [outputs, final_context_state, iterator.source_sequence_length, iterator.target_sequence_length])
    #     print("src Length:", src_len)
    #     for k, x in enumerate(p_output):
    #         print(p_output._fields[k], x.shape if isinstance(x, np.ndarray) else None)
    #         print(x)
    #     print("")
    #     print("tgt Length:", tgt_len)
    #     for k, x in enumerate(p_context):
    #         print(p_context._fields[k], x.shape if isinstance(x, np.ndarray) else None)
    #         print(x)
    #     print("----------------------")
    # saver = tf.train.Saver(tf.global_variables())
    # if not os.path.exists("tmp"):
    #     os.makedirs("tmp")
    # saver.save(sess, "tmp/sequential.ckpt")
    # sess.close()
    #
    # config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config_proto.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_proto)
    # latest_ckpt = tf.train.latest_checkpoint("tmp")
    # saver = tf.train.Saver()
    # saver.restore(sess, latest_ckpt)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.tables_initializer())
    # sess.run(iterator.initializer)
    # print(sess.run(logits).shape)
    # # ---------------------------------------------

# ------------------------------------------------------------

# Loss ------------------------------------------------------
with tf.device("/gpu:0"):
    # iterator.target_output: [batch_size, max_time]
    max_time = iterator.target_output.shape[1].value
    # tf.transpose(iterator.target_output): [max_time, batch_size]
    # logits:                               [max_time, batch_size, tgt_vocab_size]
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.transpose(iterator.target_output),
                                                              logits=logits)
    target_weights = tf.sequence_mask(iterator.target_sequence_length, max_time, dtype=logits.dtype)
    target_weights = tf.transpose(target_weights)
    # crossent:       [max_time, batch_size]
    # target_weights: [max_time, batch_size]
    train_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)

    # # Test ----------------------------------------
    # config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # config_proto.gpu_options.allow_growth = True
    # sess = tf.Session(config=config_proto)
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.tables_initializer())
    # sess.run(iterator.initializer)
    # for i in range(3):
    #     ct, tgt_weight, loss, tgt_out, tgt_len, ll = sess.run(
    #         [crossent, target_weights, train_loss, tf.transpose(iterator.target_output),
    #          iterator.target_sequence_length, logits])
    #     print(ct)
    #     print(tgt_out)
    #     print(tgt_len)
    #     print(tgt_weight)
    #     print(ll[:, 0, tgt_out[:, 0, :]])
    #     print(i, loss)
    #     print("----------------------")
    # sess.close()
    # # ---------------------------------------------


# ------------------------------------------------------


def _get_learning_rate_warmup(hparams, learning_rate, global_step):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
          (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
        # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor ** (tf.to_float(warmup_steps - global_step))
    else:
        raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        global_step < hparams.warmup_steps,
        lambda: inv_decay * learning_rate,
        lambda: learning_rate,
        name="learning_rate_warump_cond")


def _get_learning_rate_decay(hparams, learning_rate, global_step):
    """Get learning rate decay."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
        decay_factor = 0.5
        if hparams.decay_scheme == "luong5":
            start_decay_step = int(hparams.num_train_steps / 2)
            decay_times = 5
        elif hparams.decay_scheme == "luong10":
            start_decay_step = int(hparams.num_train_steps / 2)
            decay_times = 10
        elif hparams.decay_scheme == "luong234":
            start_decay_step = int(hparams.num_train_steps * 2 / 3)
            decay_times = 4
        remain_steps = hparams.num_train_steps - start_decay_step
        decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
        start_decay_step = hparams.num_train_steps
        decay_steps = 0
        decay_factor = 1.0
    elif hparams.decay_scheme:
        raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
          "decay_factor %g" % (hparams.decay_scheme, start_decay_step, decay_steps, decay_factor))

    return tf.cond(
        global_step < start_decay_step,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate,
            (global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")


# learning rate ------------------------------------------
word_count = tf.reduce_sum(iterator.source_sequence_length) + tf.reduce_sum(iterator.target_sequence_length)
predict_count = tf.reduce_sum(iterator.target_sequence_length)
global_step = tf.Variable(0, trainable=False)
params = tf.trainable_variables()

# Gradients and SGD update operation for training the model.
# Arrange for the embedding vars to appear at the beginning.
learning_rate = tf.constant(hparams.learning_rate)
# warm-up
learning_rate = _get_learning_rate_warmup(hparams, learning_rate, global_step)
# decay
learning_rate = _get_learning_rate_decay(hparams, learning_rate, global_step)
# ------------------------------------------


# Optimizer ------------------------------------------
opt = tf.train.GradientDescentOptimizer(learning_rate)
tf.summary.scalar("lr", learning_rate)
# ----------------------------------------------------


# Gradients ------------------------------------------
gradients = tf.gradients(train_loss, params, colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

clipped_grads, grad_norm_summary, grad_norm = gradient_clip(gradients, max_gradient_norm=hparams.max_gradient_norm)
grad_norm = grad_norm

update = opt.apply_gradients(zip(clipped_grads, params), global_step=global_step)
#
# # Summary
# train_summary = tf.summary.merge([tf.summary.scalar("lr", learning_rate),
#                                   tf.summary.scalar("train_loss", train_loss), ] + grad_norm_summary)
#
# # Saver
# saver = tf.train.Saver(
#     tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)
#
# # Print trainable variables
# print("# Trainable variables")
# for param in params:
#     print("  %s, %s, %s" % (param.name, str(param.get_shape()),
#                             param.op.device))
