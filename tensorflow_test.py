import tensorflow as tf
from tensorflow.python.client import device_lib


# 列出所有的本地机器设备
local_device_protos = device_lib.list_local_devices()
# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']

#查看tensorflow版本
print(tf.__version__)

print('GPU', tf.test.is_gpu_available())

local_device_protos = device_lib.list_local_devices()
# 打印
#     print(local_device_protos)

# 只打印GPU设备
[print(x) for x in local_device_protos if x.device_type == 'GPU']


# a = tf.constant(2.0)
# b = tf.constant(4.0)
# print(a + b)
