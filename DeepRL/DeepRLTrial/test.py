import tensorflow as tf

a = [[j*10+i for i in range(6)] for j in range(128)]
b = [(i % 6) for i in range(128)]

a = tf.convert_to_tensor(a, dtype=tf.float32)
b = tf.convert_to_tensor(b, dtype=tf.int16)
b = tf.expand_dims(b, axis=-1)
print(a.shape, b.shape)
print(tf.gather_nd(a, b).numpy())
# print(tf.gather_nd(a, b)[0]chatgpt)
