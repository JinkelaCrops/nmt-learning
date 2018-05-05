import tensorflow as tf
import numpy as np

# 创建输入数据,  batch为2, length为10，embedding为4
X = np.random.randn(2, 10, 4)

# 第二个example长度为6
X[1, 6:] = 0
X_lengths = [10, 6]

cell = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float64, sequence_length=X_lengths)

result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)

print(result[0])

assert result[0]["outputs"].shape == (2, 10, 3)

# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert (result[0]["outputs"][1, 7, :] == np.zeros(cell.output_size)).all()
