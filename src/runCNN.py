import os
import time
# import warnings filter

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import wandb
from tqdm import tqdm
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow as tf

# disable tf warnings
tf.get_logger().setLevel('ERROR')

# initialize wandb
# run = wandb.init(project='cnn_optimized_execution')

# set memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# loading the model
model = tf.keras.models.load_model('../res/models/my_model_tech_db_filtered2020-05-13_15-05-43_0.094_0.094.h5')

# loading the image
image = np.load('../res/dataset/1000008.npy')

# init of qp and times list
qp = 22

elapsed_times = []

# running multiple inferences
for i in tqdm(range(10000)):

	start_time = time.time()

	prediction = model.predict([[image],[qp]])

	end_time = time.time()

	elapsed_times.append((end_time - start_time) * 1000)

print('Inferences finished ! Dimensions : ', prediction.shape, ' / Average execution time : %.3f ms ' % (sum(elapsed_times)/len(elapsed_times)), sep="")

# run.finish()

plt.plot(elapsed_times)
plt.title('Evolution of inference times')
plt.xlabel('Iterations')
plt.ylabel('Execution time')

plt.show()



