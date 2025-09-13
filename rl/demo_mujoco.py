import mujoco
import numpy as np
from mujoco import viewer
import time

model_xml = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom name="box" type="box" size="0.2 0.1 0.1" rgba="0 0.8 0 1"/>
      <geom name="axle" type="cylinder" pos="0 0 0.05" size="0.02 0.05" rgba="0.8 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="slider_motor" joint="slider" gear="1"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)

# 修改后的稳定版本
with viewer.launch_passive(model, data) as v:
    # 初始化时间基准
    start_time = time.time()

    while True:
        # 计算实际流逝时间
        elapsed = time.time() - start_time

        # 更新控制信号（1Hz 正弦波）
        data.ctrl[0] = 1.0 * np.sin(2 * np.pi * elapsed)

        # 执行精确的实时仿真步进
        while data.time < elapsed:
            mujoco.mj_step(model, data)

        # 同步渲染（关键参数）
        v.sync()

        # 微小延迟防止CPU满载
        time.sleep(0.001)

# 窗口关闭后自动退出