import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# https://blog.csdn.net/weixin_57972634/article/details/147914511


def vehicle_model(x,y,psi,v,a,delta,Ts, L):
    next_x = x + Ts * v * np.cos(psi)
    next_y = y + Ts * v * np.sin(psi)
    next_psi = psi + Ts * (v / L) * np.tan(delta)
    next_v = v + a * Ts
    return next_x, next_y, next_psi, next_v

class MPC:
    def __init__(self):
        self.Ts = 0.1
        self.N = 10
        self.L = 2.5
        self.v_min, self.v_max = 0, 20
        self.a_min, self.a_max = -5, 3
        self.sigma_min, self.sigma_max = -30, 30

        self.lane1_y = 0
        self.lane2_y = 3.5
        self.target_v = 10
        self.opti = ca.Opti

    def optimize(self):





if __name__ == "__main__":



    num_frames = len(t_history)
    ani = FuncAnimation(fig, update,frames=num_frames,init_func=init, blit=True, interval=Ts*1000, repeat=False)
    animation_filename = 'NMPC_避障变道仿真动画_v6_优化边界.gif'
    try:
        ani.save(animation_filename, writer='pillow', fps=int(1 / Ts))
        print(f"动画已保存为: {animation_filename}")
    except Exception as e:
        print(f"保存动画失败: {e}")
        print("请确保已安装 Pillow (for GIF) 或 FFmpeg (for MP4)，并将其添加到系统路径。")
        print("如果问题持续，尝试在 FuncAnimation 中设置 blit=False。")

    plt.show()