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


def save_hparams(out_dir, hparams):
    """Save hparams."""
    hparams_file = os.path.join(out_dir, "hparams")
    print("  saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())


# =======================================================================
# hparams end
# =======================================================================

# =======================================================================
# model
# =======================================================================

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def _single_cell(unit_type, num_units, forget_bias, dropout, mode, residual_connection=False, device_str=None,
                 residual_fn=None):
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
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        print("  %s, dropout=%g " % (type(single_cell).__name__, dropout), end="")

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(
            single_cell, residual_fn=residual_fn)
        print("  %s" % type(single_cell).__name__, end="")

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        print("  %s, device=%s" %
              (type(single_cell).__name__, device_str), end="")

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0,
               single_cell_fn=None, residual_fn=None):
    """Create a list of RNN cells."""
    if not single_cell_fn:
        single_cell_fn = _single_cell

    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        print("  cell %d" % i, end="")
        single_cell = single_cell_fn(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=(i >= num_layers - num_residual_layers),
            device_str=get_device_str(i + base_gpu, num_gpus),
            residual_fn=residual_fn
        )
        print("")
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0,
                    single_cell_fn=None):
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
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu,
                           single_cell_fn=single_cell_fn)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _create_pretrained_emb_from_txt(vocab_file, embed_file, num_trainable_tokens=3, dtype=tf.float32, scope=None):
    """Load pretrain embeding from embed_file, and return an embedding matrix.

    Args:
      embed_file: Path to a Glove formated embedding txt file.
      num_trainable_tokens: Make the first n tokens in the vocab file as trainable
        variables. Default is 3, which is "<unk>", "<s>" and "</s>".
    """
    vocab, _ = load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]

    print("# Using pretrained embedding: %s." % embed_file)
    print("  with trainable tokens: ")

    emb_dict, emb_size = load_embed_txt(embed_file)
    for token in trainable_tokens:
        print("    %s" % token)
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size

    emb_mat = np.array(
        [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
        with tf.device(_get_embed_device(num_trainable_tokens)):
            emb_mat_var = tf.get_variable(
                "emb_mat_var", [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0)


def _create_or_load_embed(embed_name, vocab_file, embed_file, vocab_size, embed_size, dtype):
    """Create a new or load an existing embedding matrix."""
    if vocab_file and embed_file:
        embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
    else:
        with tf.device(_get_embed_device(vocab_size)):
            embedding = tf.get_variable(
                embed_name, [vocab_size, embed_size], dtype)
    return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None,
                                       src_embed_file=None,
                                       tgt_embed_file=None,
                                       scope=None):
    """Create embedding matrix for both encoder and decoder.

    Args:
      share_vocab: A boolean. Whether to share embedding matrix for both
        encoder and decoder.
      src_vocab_size: An integer. The source vocab size.
      tgt_vocab_size: An integer. The target vocab size.
      src_embed_size: An integer. The embedding dimension for the encoder's
        embedding.
      tgt_embed_size: An integer. The embedding dimension for the decoder's
        embedding.
      dtype: dtype of the embedding matrix. Default to float32.
      num_partitions: number of partitions used for the embedding vars.
      scope: VariableScope for the created subgraph. Default to "embedding".

    Returns:
      embedding_encoder: Encoder's embedding matrix.
      embedding_decoder: Decoder's embedding matrix.

    Raises:
      ValueError: if use share_vocab but source and target have different vocab
        size.
    """

    if num_partitions <= 1:
        partitioner = None
    else:
        # Note: num_partitions > 1 is required for distributed training due to
        # embedding_lookup tries to colocate single partition-ed embedding variable
        # with lookup ops. This may cause embedding variables being placed on worker
        # jobs.
        partitioner = tf.fixed_size_partitioner(num_partitions)

    if (src_embed_file or tgt_embed_file) and partitioner:
        raise ValueError(
            "Can't set num_partitions > 1 when using pretrained embedding")

    with tf.variable_scope(
            scope or "embeddings", dtype=dtype, partitioner=partitioner) as scope:
        # Share embedding
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("Share embedding but different src/tgt vocab sizes"
                                 " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
            assert src_embed_size == tgt_embed_size
            print("# Use the same embedding for source and target")
            vocab_file = src_vocab_file or tgt_vocab_file
            embed_file = src_embed_file or tgt_embed_file

            embedding_encoder = _create_or_load_embed(
                "embedding_share", vocab_file, embed_file,
                src_vocab_size, src_embed_size, dtype)
            embedding_decoder = embedding_encoder
        else:
            with tf.variable_scope("encoder", partitioner=partitioner):
                embedding_encoder = _create_or_load_embed(
                    "embedding_encoder", src_vocab_file, src_embed_file,
                    src_vocab_size, src_embed_size, dtype)

            with tf.variable_scope("decoder", partitioner=partitioner):
                embedding_decoder = _create_or_load_embed(
                    "embedding_decoder", tgt_vocab_file, tgt_embed_file,
                    tgt_vocab_size, tgt_embed_size, dtype)

    return embedding_encoder, embedding_decoder


class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass


class BaseModel(object):
    """Sequence-to-sequence base class.
    """

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None,
                 extra_args=None):
        """Create the model.

        Args:
          hparams: Hyperparameter configurations.
          mode: TRAIN | EVAL | INFER
          iterator: Dataset Iterator that feeds data.
          source_vocab_table: Lookup table mapping source words to ids.
          target_vocab_table: Lookup table mapping target words to ids.
          reverse_target_vocab_table: Lookup table mapping ids to target words. Only
            required in INFER mode. Defaults to None.
          scope: scope of the model.
          extra_args: model_helper.ExtraArgs, for passing customizable functions.

        """
        assert isinstance(iterator, BatchedInput)
        self.iterator = iterator
        self.mode = mode
        self.src_vocab_table = source_vocab_table
        self.tgt_vocab_table = target_vocab_table

        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.num_gpus = hparams.num_gpus
        self.time_major = hparams.time_major

        # extra_args: to make it flexible for adding external customizable code
        self.single_cell_fn = None
        if extra_args:
            self.single_cell_fn = extra_args.single_cell_fn

        # Set num layers
        self.num_encoder_layers = hparams.num_encoder_layers
        self.num_decoder_layers = hparams.num_decoder_layers
        assert self.num_encoder_layers
        assert self.num_decoder_layers

        # Set num residual layers
        if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
            self.num_encoder_residual_layers = hparams.num_residual_layers
            self.num_decoder_residual_layers = hparams.num_residual_layers
        else:
            self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
            self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

        # Initializer
        initializer = get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Embeddings
        self.init_embeddings(hparams, scope)
        self.batch_size = tf.size(self.iterator.source_sequence_length)

        # Projection
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(
                    hparams.tgt_vocab_size, use_bias=False, name="output_projection")

        ## Train graph
        res = self.build_graph(hparams, scope=scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
            self.word_count = tf.reduce_sum(
                self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits, _, self.final_context_state, self.sample_id = res
            self.sample_words = reverse_target_vocab_table.lookup(
                tf.to_int64(self.sample_id))

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            ## Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(
                self.iterator.target_sequence_length)

        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()

        # Gradients and SGD update operation for training the model.
        # Arrage for the embedding vars to appear at the beginning.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.learning_rate = tf.constant(hparams.learning_rate)
            # warm-up
            self.learning_rate = self._get_learning_rate_warmup(hparams)
            # decay
            self.learning_rate = self._get_learning_rate_decay(hparams)

            # Optimizer
            if hparams.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif hparams.optimizer == "adam":
                opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradients
            gradients = tf.gradients(
                self.train_loss,
                params,
                colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

            clipped_grads, grad_norm_summary, grad_norm = gradient_clip(
                gradients, max_gradient_norm=hparams.max_gradient_norm)
            self.grad_norm = grad_norm

            self.update = opt.apply_gradients(
                zip(clipped_grads, params), global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge([
                                                      tf.summary.scalar("lr", self.learning_rate),
                                                      tf.summary.scalar("train_loss", self.train_loss),
                                                  ] + grad_norm_summary)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary(hparams)

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                    param.op.device))

    def _get_learning_rate_warmup(self, hparams):
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
            inv_decay = warmup_factor ** (
                tf.to_float(warmup_steps - self.global_step))
        else:
            raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

        return tf.cond(
            self.global_step < hparams.warmup_steps,
            lambda: inv_decay * self.learning_rate,
            lambda: self.learning_rate,
            name="learning_rate_warump_cond")

    def _get_learning_rate_decay(self, hparams):
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
              "decay_factor %g" % (hparams.decay_scheme,
                                   start_decay_step,
                                   decay_steps,
                                   decay_factor))

        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def init_embeddings(self, hparams, scope):
        """Init embeddings."""
        self.embedding_encoder, self.embedding_decoder = (
            create_emb_for_encoder_and_decoder(
                share_vocab=hparams.share_vocab,
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_embed_size=hparams.num_units,
                tgt_embed_size=hparams.num_units,
                num_partitions=hparams.num_embeddings_partitions,
                src_vocab_file=hparams.src_vocab_file,
                tgt_vocab_file=hparams.tgt_vocab_file,
                src_embed_file=hparams.src_embed_file,
                tgt_embed_file=hparams.tgt_embed_file,
                scope=scope, ))

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size,
                         self.grad_norm,
                         self.learning_rate])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predict_count,
                         self.batch_size])

    def build_graph(self, hparams, scope=None):
        """Subclass must implement this method.

        Creates a sequence-to-sequence model with dynamic RNN decoder API.
        Args:
          hparams: Hyperparameter configurations.
          scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

        Returns:
          A tuple of the form (logits, loss, final_context_state),
          where:
            logits: float32 Tensor [batch_size x num_decoder_symbols].
            loss: the total loss / batch_size.
            final_context_state: The final state of decoder RNN.

        Raises:
          ValueError: if encoder_type differs from mono and bi, or
            attention_option is not (luong | scaled_luong |
            bahdanau | normed_bahdanau).
        """
        print("# creating %s graph ..." % self.mode)
        dtype = tf.float32

        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Encoder
            encoder_outputs, encoder_state = self._build_encoder(hparams)

            ## Decoder
            logits, sample_id, final_context_state = self._build_decoder(
                encoder_outputs, encoder_state, hparams)

            ## Loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device(get_device_str(self.num_encoder_layers - 1,
                                              self.num_gpus)):
                    loss = self._compute_loss(logits)
            else:
                loss = None

            return logits, loss, final_context_state, sample_id

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        """Subclass must implement this.

        Build and run an RNN encoder.

        Args:
          hparams: Hyperparameters configurations.

        Returns:
          A tuple of encoder_outputs and encoder_state.
        """
        pass

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu,
            single_cell_fn=self.single_cell_fn)

    def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
        """Maximum decoding steps at inference time."""
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
            print("  decoding maximum_iterations %d" % maximum_iterations)
        else:
            # TODO(thangluong): add decoding_length_factor flag
            decoding_length_factor = 2.0
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * decoding_length_factor))
        return maximum_iterations

    def _build_decoder(self, encoder_outputs, encoder_state, hparams):
        """Build and run a RNN decoder with a final projection layer.

        Args:
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          hparams: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                             tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                             tf.int32)
        iterator = self.iterator

        # maximum_iteration: The maximum decoding steps.
        maximum_iterations = self._get_infer_maximum_iterations(
            hparams, iterator.source_sequence_length)

        ## Decoder.
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams, encoder_outputs, encoder_state,
                iterator.source_sequence_length)

            ## Train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # decoder_emp_inp: [max_time, batch_size, num_units]
                target_input = iterator.target_input
                if self.time_major:
                    target_input = tf.transpose(target_input)
                decoder_emb_inp = tf.nn.embedding_lookup(
                    self.embedding_decoder, target_input)

                # Helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_emb_inp, iterator.target_sequence_length,
                    time_major=self.time_major)

                # Decoder
                my_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state, )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                sample_id = outputs.sample_id

                # Note: there's a subtle difference here between train and inference.
                # We could have set output_layer when create my_decoder
                #   and shared more code between train and inference.
                # We chose to apply the output_layer to all timesteps for speed:
                #   10% improvements for small models & 20% for larger ones.
                # If memory is a concern, we should apply output_layer per timestep.
                logits = self.output_layer(outputs.rnn_output)

            ## Inference
            else:
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embedding_decoder,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,
                        length_penalty_weight=length_penalty_weight)
                else:
                    # Helper
                    sampling_temperature = hparams.sampling_temperature
                    if sampling_temperature > 0.0:
                        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token,
                            softmax_temperature=sampling_temperature,
                            seed=hparams.random_seed)
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            self.embedding_decoder, start_tokens, end_token)

                    # Decoder
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell,
                        helper,
                        decoder_initial_state,
                        output_layer=self.output_layer  # applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope)

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    @abc.abstractmethod
    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Subclass must implement this.

        Args:
          hparams: Hyperparameters configurations.
          encoder_outputs: The outputs of encoder for every time step.
          encoder_state: The final state of the encoder.
          source_sequence_length: sequence length of encoder_outputs.

        Returns:
          A tuple of a multi-layer RNN cell used by decoder
            and the intial state of the decoder RNN.
        """
        pass

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self.get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
        return loss

    def _get_infer_summary(self, hparams):
        return tf.no_op()

    def infer(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
        ])

    def decode(self, sess):
        """Decode a batch.

        Args:
          sess: tensorflow session to use.

        Returns:
          A tuple consiting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        _, infer_summary, _, sample_words = self.infer(sess)

        # make sure outputs is of shape [batch_size, time] or [beam_width,
        # batch_size, time] when using beam search.
        if self.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3:  # beam search output in [batch_size,
            # time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])
        return sample_words, infer_summary


class Model(BaseModel):
    """Sequence-to-sequence dynamic model.

    This class implements a multi-layer recurrent neural network as encoder,
    and a multi-layer recurrent neural network decoder.
    """

    def _build_encoder(self, hparams):
        """Build an encoder."""
        num_layers = self.num_encoder_layers
        num_residual_layers = self.num_encoder_residual_layers
        iterator = self.iterator

        source = iterator.source
        if self.time_major:
            source = tf.transpose(source)

        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            # Look up embedding, emp_inp: [max_time, batch_size, num_units]
            encoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_encoder, source)

            # Encoder_outputs: [max_time, batch_size, num_units]
            if hparams.encoder_type == "uni":
                print("  num_layers = %d, num_residual_layers=%d" %
                      (num_layers, num_residual_layers))
                cell = self._build_encoder_cell(
                    hparams, num_layers, num_residual_layers)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    encoder_emb_inp,
                    dtype=dtype,
                    sequence_length=iterator.source_sequence_length,
                    time_major=self.time_major,
                    swap_memory=True)
            elif hparams.encoder_type == "bi":
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                print("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                      (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = (
                    self._build_bidirectional_rnn(
                        inputs=encoder_emb_inp,
                        sequence_length=iterator.source_sequence_length,
                        dtype=dtype,
                        hparams=hparams,
                        num_bi_layers=num_bi_layers,
                        num_bi_residual_layers=num_bi_residual_layers))

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
        return encoder_outputs, encoder_state

    def _build_bidirectional_rnn(self, inputs, sequence_length,
                                 dtype, hparams,
                                 num_bi_layers,
                                 num_bi_residual_layers,
                                 base_gpu=0):
        """Create and call biddirectional RNN cells.

        Args:
          num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
            layers in each RNN cell will be wrapped with `ResidualWrapper`.
          base_gpu: The gpu device id to use for the first forward RNN layer. The
            i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
            device id. The `base_gpu` for backward RNN cell is `(base_gpu +
            num_bi_layers)`.

        Returns:
          The concatenated bidirectional output and the bidirectional RNN cell"s
          state.
        """
        # Construct forward and backward cells
        fw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=base_gpu)
        bw_cell = self._build_encoder_cell(hparams,
                                           num_bi_layers,
                                           num_bi_residual_layers,
                                           base_gpu=(base_gpu + num_bi_layers))

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build an RNN cell that can be used by decoder."""
        # We only make use of encoder_outputs in attention-based models
        if hparams.attention:
            raise ValueError("BasicModel doesn't support attention.")

        cell = create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=self.num_decoder_layers,
            num_residual_layers=self.num_decoder_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=self.num_gpus,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn)

        # For beam search, we need to replicate encoder infos beam_width times
        if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=hparams.beam_width)
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state


class AttentionModel(Model):
    """Sequence-to-sequence dynamic model with attention.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
    This class also allows to use GRU cells in addition to LSTM cells with
    support for dropout.
    """

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None,
                 scope=None,
                 extra_args=None):
        # Set attention_mechanism_fn
        if extra_args and extra_args.attention_mechanism_fn:
            self.attention_mechanism_fn = extra_args.attention_mechanism_fn
        else:
            self.attention_mechanism_fn = create_attention_mechanism

        super(AttentionModel, self).__init__(
            hparams=hparams,
            mode=mode,
            iterator=iterator,
            source_vocab_table=source_vocab_table,
            target_vocab_table=target_vocab_table,
            reverse_target_vocab_table=reverse_target_vocab_table,
            scope=scope,
            extra_args=extra_args)

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_summary = self._get_infer_summary(hparams)

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        attention_option = hparams.attention
        attention_architecture = hparams.attention_architecture

        if attention_architecture != "standard":
            raise ValueError(
                "Unknown attention architecture %s" % attention_architecture)

        num_units = hparams.num_units
        num_layers = self.num_decoder_layers
        num_residual_layers = self.num_decoder_residual_layers
        beam_width = hparams.beam_width

        dtype = tf.float32

        # Ensure memory is batch-major
        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = self.attention_mechanism_fn(
            attention_option, num_units, memory, source_sequence_length, self.mode)

        cell = create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=self.num_gpus,
            mode=self.mode,
            single_cell_fn=self.single_cell_fn)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                             beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            output_attention=hparams.output_attention,
            name="attention")

        # TODO(thangluong): do we need num_layers, num_gpus?
        cell = tf.contrib.rnn.DeviceWrapper(cell, get_device_str(num_layers - 1, self.num_gpus))

        if hparams.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state

    def _get_infer_summary(self, hparams):
        if hparams.beam_width > 0:
            return tf.no_op()
        return _create_attention_images_summary(self.final_context_state)


def create_attention_mechanism(attention_option, num_units, memory, source_sequence_length, mode):
    """Create attention mechanism based on the attention_option."""
    del mode  # unused

    # Mechanism
    if attention_option == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def _create_attention_images_summary(final_context_state):
    """create attention image and attention summary."""
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(
        tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary


# =======================================================================
# model end
# =======================================================================

# ===================================================================
# load model start
# ===================================================================

def get_infer_iterator(src_dataset, src_vocab_table, batch_size, eos, src_max_len=None):
    # infer model的数据流
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
        src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
    # Add in the word counts.
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                0))  # src_len -- unused

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None)


def get_iterator(src_dataset, tgt_dataset,
                 src_vocab_table, tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)  # 转换 https://tensorflow.google.cn/versions/r1.7/api_docs/python/tf/cast
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (
            src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
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

    if num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
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

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
     tgt_seq_len) = (batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)
# TODO: here

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator", "skip_count_placeholder"))):
    pass


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    src_vocab_table = lookup_ops.index_table_from_file(
        src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
            tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table


def create_train_model(model_creator, hparams, scope=None, num_workers=1, jobid=0, extra_args=None):
    """Create train graph, model, and iterator."""
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)  # 训练src
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)  # 训练tgt
    src_vocab_file = hparams.src_vocab_file  # 训练src vocab
    tgt_vocab_file = hparams.tgt_vocab_file  # 训练tgt vocab

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, hparams.share_vocab)  # 2个vocab, https://tensorflow.google.cn/versions/r1.7/api_docs/python/tf/contrib/lookup/index_table_from_file

        src_dataset = tf.data.TextLineDataset(src_file)  # https://tensorflow.google.cn/versions/r1.7/api_docs/python/tf/data/TextLineDataset
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        # todo: here
        iterator = get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder,
            num_shards=num_workers,
            shard_index=jobid)

        # Note: One can set model_device_fn to
        # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
        model_device_fn = None
        if extra_args: model_device_fn = extra_args.model_device_fn
        with tf.device(model_device_fn):
            model = model_creator(
                hparams,
                iterator=iterator,
                mode=tf.contrib.learn.ModeKeys.TRAIN,
                source_vocab_table=src_vocab_table,
                target_vocab_table=tgt_vocab_table,
                scope=scope,
                extra_args=extra_args)

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)


class EvalModel(collections.namedtuple("EvalModel",
                                       ("graph", "model", "src_file_placeholder", "tgt_file_placeholder", "iterator"))):
    pass


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
    """Create train graph, model, src/tgt file holders, and iterator."""
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file
    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "eval"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        src_dataset = tf.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
        iterator = get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return EvalModel(
        graph=graph,
        model=model,
        src_file_placeholder=src_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder,
        iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel", ("graph", "model", "src_placeholder", "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
    """Create inference model."""
    graph = tf.Graph()
    src_vocab_file = hparams.src_vocab_file
    tgt_vocab_file = hparams.tgt_vocab_file

    with graph.as_default(), tf.container(scope or "infer"):
        src_vocab_table, tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, hparams.share_vocab)
        reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
            tgt_vocab_file, default_value=UNK)

        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        src_dataset = tf.data.Dataset.from_tensor_slices(
            src_placeholder)
        iterator = get_infer_iterator(
            src_dataset,
            src_vocab_table,
            batch_size=batch_size_placeholder,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len_infer)
        model = model_creator(
            hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            source_vocab_table=src_vocab_table,
            target_vocab_table=tgt_vocab_table,
            reverse_target_vocab_table=reverse_tgt_vocab_table,
            scope=scope,
            extra_args=extra_args)
    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)


def load_data(inference_input_file, hparams=None):
    """Load inference data."""
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(inference_input_file, mode="rb")) as f:
        inference_data = f.read().splitlines()

    if hparams and hparams.inference_indices:
        inference_data = [inference_data[i] for i in hparams.inference_indices]

    return inference_data


def get_config_proto(log_device_placement=False, allow_soft_placement=True, num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,  # 记录设备指派情况 , 可以看到/job:localhost/replica:0/task:0/device:GPU:0
        allow_soft_placement=allow_soft_placement)  # 自动选择一个存在并且支持的设备来运行
    config_proto.gpu_options.allow_growth = True  # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加

    # CPU threads options, 不使用GPU
    # intra_op_parallelism_threads 控制运算符op内部的并行
    # 当运算符op为单一运算符，并且内部可以实现并行时，如矩阵乘法，reduce_sum之类的操作，
    # 可以通过设置intra_op_parallelism_threads参数来并行, intra代表内部。
    # inter_op_parallelism_threads 控制多个运算符op之间的并行计算.
    # 当有多个运算符op，并且他们之间比较独立，运算符和运算符之间没有直接的路径Path相连。
    # Tensorflow会尝试并行地计算他们，使用由inter_op_parallelism_threads参数来控制数量的一个线程池。
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("  loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)  # 加载最近的一个模型
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())  # 初始化，在之前load了model
        session.run(tf.tables_initializer())  # 初始化table, table是什么? 应该是vocab字典 string -> index
        print("  created %s model with fresh parameters, time %.2fs" % (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)  # 训练轮数
    return model, global_step


# ===================================================================
# load model end
# ===================================================================

# ===================================================================
# summary and eval
# ===================================================================

def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def compute_perplexity(model, sess, name):
    """Compute perplexity of the output of the model.

    Args:
      model: model for compute perplexity.
      sess: tensorflow session to use.
      name: name of the batch.

    Returns:
      The perplexity of the eval outputs.
    """
    total_loss = 0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            loss, predict_count, batch_size = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break

    perplexity = safe_exp(total_loss / total_predict_count)
    print("  eval %s: perplexity %.2f" % (name, perplexity), start_time)
    return perplexity


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict, summary_writer, label):
    """Computing perplexity."""
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    ppl = compute_perplexity(model, sess, label)
    add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
    return ppl


def run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer, use_test_set=True):
    """Compute internal evaluation (perplexity) for both dev / test."""
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = create_or_load_model(eval_model.model, model_dir, eval_sess, "eval")

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: dev_src_file,
        eval_model.tgt_file_placeholder: dev_tgt_file
    }

    dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                             eval_model.iterator, dev_eval_iterator_feed_dict,
                             summary_writer, "dev")
    test_ppl = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: test_src_file,
            eval_model.tgt_file_placeholder: test_tgt_file
        }
        test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                                  eval_model.iterator, test_eval_iterator_feed_dict,
                                  summary_writer, "test")
    return dev_ppl, test_ppl


def _format_results(name, ppl, scores, metrics):
    """Format results."""
    result_str = ""
    if ppl:
        result_str = "%s ppl %.2f" % (name, ppl)
    if scores:
        for metric in metrics:
            if result_str:
                result_str += ", %s %s %.1f" % (name, metric, scores[metric])
            else:
                result_str = "%s %s %.1f" % (name, metric, scores[metric])
    return result_str


def format_text(words):
    """Convert a sequence words into sentence."""
    if (not hasattr(words, "__len__") and  # for numpy array
            not isinstance(words, collections.Iterable)):
        words = [words]
    return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = b""
    if isinstance(symbols, str):
        symbols = symbols.encode()
    delimiter_len = len(delimiter)
    for symbol in symbols:
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            word += symbol[:-delimiter_len]
        else:  # end of a word
            word += symbol
            words.append(word)
            word = b""
    return b" ".join(words)


def format_spm_text(symbols):
    """Decode a text in SPM (https://github.com/google/sentencepiece) format."""
    return u"".join(format_text(symbols).decode("utf-8").split()).replace(
        u"\u2581", u" ").strip().encode("utf-8")


def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    if subword_option == "bpe":  # BPE
        translation = format_bpe_text(output)
    elif subword_option == "spm":  # SPM
        translation = format_spm_text(output)
    else:
        translation = format_text(output)

    return translation


def evaluate(ref_file, trans_file, metric, subword_option=None):
    """Pick a metric and evaluate depending on task."""
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, trans_file,
                                 subword_option=subword_option)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ref_file, trans_file,
                                  subword_option=subword_option)
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, trans_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, trans_file)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, subword_option):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()

    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)

    # SPM
    elif subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, subword_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line, subword_option=None)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
        for line in fh:
            references.append(_clean(line, subword_option))

    hypotheses = []
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(summarization_file, "rb")) as fh:
        for line in fh:
            hypotheses.append(_clean(line, subword_option=None))

    rouge_score_map = rouge.rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
    """Compute accuracy, each line contains a label."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = label.strip()
                pred = pred_fh.readline().strip()
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True):
    """Decode a test set and compute a score according to the evaluation task."""
    # Decode
    if decode:
        print("  decoding to output %s." % trans_file)

        start_time = time.time()
        num_sentences = 0
        with codecs.getwriter("utf-8")(
                tf.gfile.GFile(trans_file, mode="wb")) as trans_f:
            trans_f.write("")  # Write empty string to ensure file is created.

            num_translations_per_input = max(
                min(num_translations_per_input, beam_width), 1)
            while True:
                try:
                    nmt_outputs, _ = model.decode(sess)
                    if beam_width == 0:
                        nmt_outputs = np.expand_dims(nmt_outputs, 0)

                    batch_size = nmt_outputs.shape[1]
                    num_sentences += batch_size

                    for sent_id in range(batch_size):
                        for beam_id in range(num_translations_per_input):
                            translation = get_translation(
                                nmt_outputs[beam_id],
                                sent_id,
                                tgt_eos=tgt_eos,
                                subword_option=subword_option)
                            trans_f.write((translation + b"\n").decode("utf-8"))
                except tf.errors.OutOfRangeError:
                    print(
                        "  done, num sentences %d, num translations per input %d" %
                        (num_sentences, num_translations_per_input), start_time)
                    break

    # Evaluation
    evaluation_scores = {}
    if ref_file and tf.gfile.Exists(trans_file):
        for metric in metrics:
            score = evaluate(
                ref_file,
                trans_file,
                metric,
                subword_option=subword_option)
            evaluation_scores[metric] = score
            print("  %s %s: %.1f" % (metric, name, score))

    return evaluation_scores


def _external_eval(model, global_step, sess, hparams, iterator, iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best, avg_ckpts=False):
    """External evaluation such as BLEU and ROUGE scores."""
    out_dir = hparams.out_dir
    decode = global_step > 0

    if avg_ckpts:
        label = "avg_" + label

    if decode:
        print("# External evaluation, global step %d" % global_step)

    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)

    output = os.path.join(out_dir, "output_%s" % label)
    scores = decode_and_evaluate(
        label,
        model,
        sess,
        output,
        ref_file=tgt_file,
        metrics=hparams.metrics,
        subword_option=hparams.subword_option,
        beam_width=hparams.beam_width,
        tgt_eos=hparams.eos,
        decode=decode)
    # Save on best metrics
    if decode:
        for metric in hparams.metrics:
            if avg_ckpts:
                best_metric_label = "avg_best_" + metric
            else:
                best_metric_label = "best_" + metric

            add_summary(summary_writer, global_step, "%s_%s" % (label, metric), scores[metric])
            # metric: larger is better
            if save_on_best and scores[metric] > getattr(hparams, best_metric_label):
                setattr(hparams, best_metric_label, scores[metric])
                model.saver.save(
                    sess,
                    os.path.join(
                        getattr(hparams, best_metric_label + "_dir"), "translate.ckpt"),
                    global_step=model.global_step)
        save_hparams(out_dir, hparams)
    return scores


def run_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, save_best_dev=True,
                      use_test_set=True, avg_ckpts=False):
    """Compute external evaluation (bleu, rouge, etc.) for both dev / test."""
    with infer_model.graph.as_default():
        loaded_infer_model, global_step = create_or_load_model(
            infer_model.model, model_dir, infer_sess, "infer")

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    dev_infer_iterator_feed_dict = {
        infer_model.src_placeholder: load_data(dev_src_file),
        infer_model.batch_size_placeholder: hparams.infer_batch_size,
    }
    dev_scores = _external_eval(
        loaded_infer_model,
        global_step,
        infer_sess,
        hparams,
        infer_model.iterator,
        dev_infer_iterator_feed_dict,
        dev_tgt_file,
        "dev",
        summary_writer,
        save_on_best=save_best_dev,
        avg_ckpts=avg_ckpts)

    test_scores = None
    if use_test_set and hparams.test_prefix:
        test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
        test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
        test_infer_iterator_feed_dict = {
            infer_model.src_placeholder: load_data(test_src_file),
            infer_model.batch_size_placeholder: hparams.infer_batch_size,
        }
        test_scores = _external_eval(
            loaded_infer_model,
            global_step,
            infer_sess,
            hparams,
            infer_model.iterator,
            test_infer_iterator_feed_dict,
            test_tgt_file,
            "test",
            summary_writer,
            save_on_best=False,
            avg_ckpts=avg_ckpts)
    return dev_scores, test_scores, global_step


def avg_checkpoints(model_dir, num_last_checkpoints, global_step, global_step_name):
    """Average the last N checkpoints in the model_dir."""
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    if not checkpoint_state:
        print("# No checkpoint file found in directory: %s" % model_dir)
        return None

    # Checkpoints are ordered from oldest to newest.
    checkpoints = (
        checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

    if len(checkpoints) < num_last_checkpoints:
        print(
            "# Skipping averaging checkpoints because not enough checkpoints is "
            "avaliable."
        )
        return None

    avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
    if not tf.gfile.Exists(avg_model_dir):
        print(
            "# Creating new directory %s for saving averaged checkpoints." %
            avg_model_dir)
        tf.gfile.MakeDirs(avg_model_dir)

    print("# Reading and averaging variables in checkpoints:")
    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if name != global_step_name:
            var_values[name] = np.zeros(shape)

    for checkpoint in checkpoints:
        print("    %s" % checkpoint)
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor

    for name in var_values:
        var_values[name] /= len(checkpoints)

    # Build a graph with same variables in the checkpoints, and save the averaged
    # variables into the avg_model_dir.
    with tf.Graph().as_default():
        tf_vars = [
            tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
            for v in var_values
        ]

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        global_step_var = tf.Variable(
            global_step, name=global_step_name, trainable=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                                   six.iteritems(var_values)):
                sess.run(assign_op, {p: value})

            # Use the built saver to save the averaged checkpoint. Only keep 1
            # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
            saver.save(
                sess,
                os.path.join(avg_model_dir, "translate.ckpt"))

    return avg_model_dir


def run_avg_external_eval(infer_model, infer_sess, model_dir, hparams, summary_writer, global_step):
    """Creates an averaged checkpoint and run external eval with it."""
    avg_dev_scores, avg_test_scores = None, None
    if hparams.avg_ckpts:
        # Convert VariableName:0 to VariableName.
        global_step_name = infer_model.model.global_step.name.split(":")[0]
        avg_model_dir = avg_checkpoints(
            model_dir, hparams.num_keep_ckpts, global_step, global_step_name)

        if avg_model_dir:
            avg_dev_scores, avg_test_scores, _ = run_external_eval(
                infer_model,
                infer_sess,
                avg_model_dir,
                hparams,
                summary_writer,
                avg_ckpts=True)

    return avg_dev_scores, avg_test_scores


def run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer, src_data, tgt_data):
    """Sample decode a random sentence from src_data."""
    with infer_model.graph.as_default():
        # infer_model load
        loaded_infer_model, global_step = create_or_load_model(infer_model.model, model_dir, infer_sess, "infer")

    decode_id = random.randint(0, len(src_data) - 1)  # 随机从src_data选一句
    print("  # %d" % decode_id)

    iterator_feed_dict = {
        infer_model.src_placeholder: [src_data[decode_id]],  # single sentence
        infer_model.batch_size_placeholder: 1,  # feed batch size = 1
    }
    infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)

    nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)

    if hparams.beam_width > 0:
        # get the top translation.
        nmt_outputs = nmt_outputs[0]

    translation = get_translation(
        nmt_outputs,
        sent_id=0,
        tgt_eos=hparams.eos,
        subword_option=hparams.subword_option)
    print("    src: %s" % src_data[decode_id])
    print("    ref: %s" % tgt_data[decode_id])
    print(b"    nmt: " + translation)

    # Summary
    if attention_summary is not None:
        summary_writer.add_summary(attention_summary, global_step)


def run_full_eval(model_dir,
                  infer_model, infer_sess,
                  eval_model, eval_sess,
                  hparams,
                  summary_writer,
                  sample_src_data, sample_tgt_data,
                  avg_ckpts=False):
    """Wrapper for running sample_decode, internal_eval and external_eval."""
    run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                      sample_src_data, sample_tgt_data)
    dev_ppl, test_ppl = run_internal_eval(
        eval_model, eval_sess, model_dir, hparams, summary_writer)
    dev_scores, test_scores, global_step = run_external_eval(
        infer_model, infer_sess, model_dir, hparams, summary_writer)

    metrics = {
        "dev_ppl": dev_ppl,
        "test_ppl": test_ppl,
        "dev_scores": dev_scores,
        "test_scores": test_scores,
    }

    avg_dev_scores, avg_test_scores = None, None
    if avg_ckpts:
        avg_dev_scores, avg_test_scores = run_avg_external_eval(
            infer_model, infer_sess, model_dir, hparams, summary_writer,
            global_step)
        metrics["avg_dev_scores"] = avg_dev_scores
        metrics["avg_test_scores"] = avg_test_scores

    result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
    if avg_dev_scores:
        result_summary += ", " + _format_results("avg_dev", None, avg_dev_scores,
                                                 hparams.metrics)
    if hparams.test_prefix:
        result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                                 hparams.metrics)
        if avg_test_scores:
            result_summary += ", " + _format_results("avg_test", None,
                                                     avg_test_scores, hparams.metrics)

    return result_summary, global_step, metrics


def process_stats(stats, info, global_step, steps_per_stats, log_f):
    """Update info and check for overflow."""
    # Update info
    info["avg_step_time"] = stats["step_time"] / steps_per_stats
    info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
    info["train_ppl"] = safe_exp(stats["loss"] / stats["predict_count"])
    info["speed"] = stats["total_count"] / (1000 * stats["step_time"])

    # Check for overflow
    is_overflow = False
    train_ppl = info["train_ppl"]
    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
        print("  step %d overflow, stop early" % global_step, log_f)
        is_overflow = True

    return is_overflow


def print_step_info(prefix, global_step, info, result_summary, log_f):
    """Print all info at the current global step."""
    print(
        "%sstep %d lr %g step-time %.2fs wps %.2fK ppl %.2f gN %.2f %s, %s" %
        (prefix, global_step, info["learning_rate"], info["avg_step_time"],
         info["speed"], info["train_ppl"], info["avg_grad_norm"], result_summary,
         time.ctime()),
        log_f)


def _get_best_results(hparams):
    """Summary of the current best results."""
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)


# ===========================================================
# summary and eval end
# ===========================================================

# make hparams ===================================================================
my_args = """nmt.nmt \
--src=de --tgt=en \
--vocab_prefix=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/vocab.bpe.32000  \
--train_prefix=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/train.tok.bpe.32000 \
--dev_prefix=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/newstest2016.tok.bpe.32000  \
--test_prefix=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16_de_en/newstest2015.tok.bpe.32000 \
--out_dir=/media/tmxmall/a36811aa-0e87-4ba1-b14f-370134452449/wmt16/wmt16-model-single \
--attention=luong \
--num_intra_threads=5 \
--num_inter_threads=1 \
--num_train_steps=3 \
--steps_per_stats=100 \
--num_layers=4 \
--num_units=1024 \
--dropout=0.2 \
--metrics=bleu
""".replace("=", " ").split()

nmt_parser = argparse.ArgumentParser()  # 创建args parser
add_arguments(nmt_parser)  # 添加需要的参数选项
flags, unparsed = nmt_parser.parse_known_args(my_args)  # 传递参数，排除冲突

default_hparams = create_hparams(flags)  # 转为tf的hparams
hparams = extend_hparams(default_hparams)

# train, load model ===================================================================
out_dir = flags.out_dir
if not os.path.exists(out_dir): os.makedirs(out_dir)

scope = ""
target_session = ""

log_device_placement = hparams.log_device_placement  # 記錄設備指派情況，可以看到 /job:localhost/replica:0/task:0/device:GPU:0
out_dir = hparams.out_dir
num_train_steps = hparams.num_train_steps
steps_per_stats = hparams.steps_per_stats
steps_per_external_eval = hparams.steps_per_external_eval  # test
steps_per_eval = 10 * steps_per_stats  # 每10个统计就valid
avg_ckpts = hparams.avg_ckpts  # ckpts平均统计

if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval  # 每50个统计就test

model_creator = AttentionModel  # 使用attention, 可以选择的有gnmt_model.GNMTModel, nmt.Model, attention.AttentionModel
# train, eval, infer三個不同的模型
train_model = create_train_model(model_creator, hparams, scope)
eval_model = create_eval_model(model_creator, hparams, scope)
infer_model = create_infer_model(model_creator, hparams, scope)

# Preload data for sample decoding. test数据
dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
sample_src_data = load_data(dev_src_file)
sample_tgt_data = load_data(dev_tgt_file)

summary_name = "train_log"
model_dir = hparams.out_dir
# Log and output files
log_file = os.path.join(out_dir, "log_%d" % time.time())
log_f = tf.gfile.GFile(log_file, mode="a")  # 写入日志文件，追加
print("# log_file=%s" % log_file, log_f)

# tf.Session 选项
config_proto = get_config_proto(
    log_device_placement=log_device_placement,  # 记录设备指派情况 , 可以看到/job:localhost/replica:0/task:0/device:GPU:0
    num_intra_threads=hparams.num_intra_threads,  # cpu training
    num_inter_threads=hparams.num_inter_threads)  # cpu training
# TensorFlow model, 3个graph的地址不同
train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)
eval_sess = tf.Session(target=target_session, config=config_proto, graph=eval_model.graph)
infer_sess = tf.Session(target=target_session, config=config_proto, graph=infer_model.graph)

with train_model.graph.as_default():
    loaded_train_model, global_step = create_or_load_model(train_model.model, model_dir, train_sess, "train")  # 初始化

# Summary writer, 记录
summary_writer = tf.summary.FileWriter(os.path.join(out_dir, summary_name), train_model.graph)

# # First evaluation, test and valid(dev)
# run_full_eval(model_dir,
#               infer_model, infer_sess,
#               eval_model, eval_sess,
#               hparams,
#               summary_writer,
#               sample_src_data, sample_tgt_data,
#               avg_ckpts)

last_stats_step = global_step
last_eval_step = global_step
last_external_eval_step = global_step

# This is the training loop.
"""Misc tasks to do before training."""
stats = {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0, "total_count": 0.0, "grad_norm": 0.0}
info = {"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
        "avg_grad_norm": 0.0,
        "learning_rate": loaded_train_model.learning_rate.eval(
            session=train_sess)}
start_train_time = time.time()
print("# Start step %d, lr %g, %s" %
      (global_step, info["learning_rate"], time.ctime()), log_f)
# Initialize all of the iterators
skip_count = hparams.batch_size * hparams.epoch_step
print("# Init train iterator, skipping %d elements" % skip_count)
train_sess.run(
    train_model.iterator.initializer,
    feed_dict={train_model.skip_count_placeholder: skip_count})

while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
        step_result = loaded_train_model.train(train_sess)
        hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
        # Finished going through the training dataset.  Go to next epoch.
        hparams.epoch_step = 0
        print(
            "# Finished an epoch, step %d. Perform external evaluation" %
            global_step)
        run_sample_decode(infer_model, infer_sess, model_dir, hparams,
                          summary_writer, sample_src_data, sample_tgt_data)
        run_external_eval(infer_model, infer_sess, model_dir, hparams,
                          summary_writer)

        if avg_ckpts:
            run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                                  summary_writer, global_step)

        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={train_model.skip_count_placeholder: 0})
        continue

    # Process step_result, accumulate stats, and write summary
    """Update stats: write summary and accumulate statistics."""
    (_, step_loss, step_predict_count, step_summary, global_step,
     step_word_count, batch_size, grad_norm, learning_rate) = step_result
    # Update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)
    stats["grad_norm"] += grad_norm
    info["learning_rate"] = learning_rate

    summary_writer.add_summary(step_summary, global_step)

    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
        last_stats_step = global_step
        is_overflow = process_stats(
            stats, info, global_step, steps_per_stats, log_f)
        print_step_info("  ", global_step, info, _get_best_results(hparams),
                        log_f)
        if is_overflow:
            break

        # Reset statistics
        stats = {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0, "total_count": 0.0, "grad_norm": 0.0}

    if global_step - last_eval_step >= steps_per_eval:
        last_eval_step = global_step
        print("# Save eval, global step %d" % global_step)
        add_summary(summary_writer, global_step, "train_ppl",
                    info["train_ppl"])

        # Save checkpoint
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(out_dir, "translate.ckpt"),
            global_step=global_step)

        # Evaluate on dev/test
        run_sample_decode(infer_model, infer_sess,
                          model_dir, hparams, summary_writer, sample_src_data,
                          sample_tgt_data)
        run_internal_eval(
            eval_model, eval_sess, model_dir, hparams, summary_writer)

    if global_step - last_external_eval_step >= steps_per_external_eval:
        last_external_eval_step = global_step

        # Save checkpoint
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(out_dir, "translate.ckpt"),
            global_step=global_step)
        run_sample_decode(infer_model, infer_sess,
                          model_dir, hparams, summary_writer, sample_src_data,
                          sample_tgt_data)
        run_external_eval(
            infer_model, infer_sess, model_dir,
            hparams, summary_writer)

        if avg_ckpts:
            run_avg_external_eval(infer_model, infer_sess, model_dir, hparams,
                                  summary_writer, global_step)

# Done training
loaded_train_model.saver.save(
    train_sess,
    os.path.join(out_dir, "translate.ckpt"),
    global_step=global_step)

result_summary, _, final_eval_metrics = run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                                                      hparams, summary_writer, sample_src_data, sample_tgt_data,
                                                      avg_ckpts)
print_step_info("# Final, ", global_step, info, result_summary, log_f)
print("# Done training!", start_train_time)

summary_writer.close()

# print("# Start evaluating saved best models.")
# for metric in hparams.metrics:
#     best_model_dir = getattr(hparams, "best_" + metric + "_dir")
#     summary_writer = tf.summary.FileWriter(
#         os.path.join(best_model_dir, summary_name), infer_model.graph)
#     result_summary, best_global_step, _ = run_full_eval(
#         best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
#         summary_writer, sample_src_data, sample_tgt_data)
#     print_step_info("# Best %s, " % metric, best_global_step, info,
#                     result_summary, log_f)
#     summary_writer.close()
#
#     if avg_ckpts:
#         best_model_dir = getattr(hparams, "avg_best_" + metric + "_dir")
#         summary_writer = tf.summary.FileWriter(
#             os.path.join(best_model_dir, summary_name), infer_model.graph)
#         result_summary, best_global_step, _ = run_full_eval(
#             best_model_dir, infer_model, infer_sess, eval_model, eval_sess,
#             hparams, summary_writer, sample_src_data, sample_tgt_data)
#         print_step_info("# Averaged Best %s, " % metric, best_global_step, info,
#                         result_summary, log_f)
#         summary_writer.close()
