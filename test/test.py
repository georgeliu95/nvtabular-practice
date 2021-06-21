import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
def step_fn():
    ctx = tf.distribute.get_replica_context()
    value = tf.identity(1.)
    print(value)
    return ctx.all_reduce("SUM", value)
print(strategy.experimental_local_results(strategy.run(step_fn)))