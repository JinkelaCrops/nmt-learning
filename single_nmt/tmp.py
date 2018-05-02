a = sess.graph.get_operations()[0]
t1 = a.outputs[0]
t1a = sess.graph.get_tensor_by_name(t1.name)
sess.run(t1a)

import re

re.sub("\u0020(\u0020?)", "\u00a0\\1", "A     Z")
