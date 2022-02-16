from safe_gpu import safe_gpu


############################################################
# Start up TensorFlow & the GPU
############################################################
gpu_owner = safe_gpu.GPUOwner(placeholder_fn=safe_gpu.tensorflow_placeholder)


log.info(f"TensorFlow version: {tf.__version__}")
log.info(f"Physical Devices: {tf.config.list_physical_devices('GPU')}")


tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_session = tf.compat.v1.Session(config=tf_config)
