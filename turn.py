from tensorflow import keras
from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel, SpatialAttention

model = keras.models.load_model(
    'dueling.h5',
    custom_objects={'SpatialAttention': SpatialAttention}
)


# 儲存為 .keras 格式（更安全，推薦）
model.save('dueling.keras')
