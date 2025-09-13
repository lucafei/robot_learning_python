import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, splev

def bernstein_poly(n, i, t):
    '''
    t为参数
    :param n:
    :param i:
    :param t:
    :return:
    '''
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier(t, control_points):
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def generate_bezier_curve(control_points, n_points=100):
    """
    生成贝塞尔曲线
    :param control_points: 控制点数组，形状为 (n, 2)
    :param n_points: 插值点数量
    :return: 贝塞尔曲线上的点
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))
    return np.array(traj)

def generate_bezier_curve_segments(control_points, n_points=100):
    """
    生成贝塞尔曲线，根据路径的几何特征动态调整控制点分布
    :param control_points: 控制点数组，形状为 (n, 2)
    :param n_points: 插值点数量
    :return: 拼接后的贝塞尔曲线
    """
    if len(control_points) < 2:
        return generate_bezier_curve(control_points, n_points)

    trajectory = []
    i = 0
    while i < len(control_points) - 1:
        if i < len(control_points) - 2 and is_corner(control_points[i], control_points[i+1], control_points[i+2]):
            start_idx = i
            while start_idx > 0 and not is_corner(control_points[start_idx-1], control_points[start_idx], control_points[start_idx+1]):
                start_idx -= 1
            end_idx = i + 2
            while end_idx < len(control_points) - 2 and is_corner(control_points[end_idx], control_points[end_idx+1], control_points[end_idx+2]):
                end_idx += 1
            start_idx = max(0, start_idx - 1)
            end_idx = min(len(control_points) - 1, end_idx + 1)
            segment_points = control_points[start_idx:end_idx+1]
            segment_traj = generate_bezier_curve(segment_points, n_points)
            trajectory.append(segment_traj)
            i = end_idx
        else:
            end_idx = i + 1
            while end_idx < len(control_points) - 2 and not is_corner(control_points[end_idx-1], control_points[end_idx], control_points[end_idx+1]):
                end_idx += 1
            segment_points = control_points[i:end_idx+1]
            if len(segment_points) < 3:
                segment_points = control_points[i:i+3] if i+3 <= len(control_points) else control_points[i:]
            segment_traj = generate_bezier_curve(segment_points, n_points)
            trajectory.append(segment_traj)
            i = end_idx

    trajectory = np.concatenate(trajectory, axis=0)
    return trajectory

def is_corner(p0, p1, p2):
    vector1 = (p1[0] - p0[0], p1[1] - p0[1])
    vector2 = (p2[0] - p1[0], p2[1] - p1[1])
    return vector1[1] * vector2[0] != vector1[0] * vector2[1]

def plot_curve(input_image_path, traj, control_points, obstacle_mask=None):
    img = plt.imread(input_image_path)
    plt.imshow(img)
    plt.plot(traj[:, 0], traj[:, 1], 'r-', label='Bezier Curve')
    plt.plot([p[0] for p in control_points], [p[1] for p in control_points], 'bo-', markersize=5, label='Control Points')
    if obstacle_mask is not None:
        obstacle_y, obstacle_x = np.where(obstacle_mask)
        plt.scatter(obstacle_x, obstacle_y, c='k', s=1, alpha=0.5, label='Obstacles')
    plt.title('Generated Bezier Curve')
    plt.legend()
    plt.show()

def smooth_path_with_bspline(path, degree=5, num_points=100):
    x = path[:, 0]
    y = path[:, 1]
    t = np.linspace(0, 1, len(x))
    spline_x = make_interp_spline(t, x, k=degree)
    spline_y = make_interp_spline(t, y, k=degree)
    t_new = np.linspace(0, 1, num_points)
    x_new = spline_x(t_new)
    y_new = spline_y(t_new)
    return np.column_stack((x_new, y_new))

def check_curve_with_obstacles(curve, obstacle_mask):
    for point in curve:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < obstacle_mask.shape[1] and 0 <= y < obstacle_mask.shape[0]:
            if obstacle_mask[y, x]:
                return True
    return False

def adjust_control_points(control_points, obstacle_mask):
    return control_points

def calculate_euclidean_distance(points):
    """
    计算路径的欧式距离
    :param points: 路径点数组
    :return: 欧式距离
    """
    distance = 0.0
    for i in range(1, len(points)):
        distance += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
    return distance


def plot_curve_with_distance(input_image_path, original_path, dilated_path, bezier_curve_points, dilated_obstacle_mask, output_filename):
    """
    绘制并保存包含三条曲线及其欧式距离的左右布局图
    :param input_image_path: 输入图像路径
    :param original_path: 原始 A* 路径
    :param dilated_path: 膨胀后的 A* 路径
    :param bezier_curve_points: 贝塞尔曲线点
    :param dilated_obstacle_mask: 膨胀后的障碍物掩码
    :param output_filename: 输出文件名
    """
    img = plt.imread(input_image_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 左边：A* 初始路径（膨胀后的和原始的）
    ax1.imshow(img)
    ax1.plot([p[0] for p in original_path], [p[1] for p in original_path], 'b--', markersize=5, label='Original A* Path')
    ax1.plot([p[0] for p in dilated_path], [p[1] for p in dilated_path], 'b-', markersize=5, label='Dilated A* Path')
    ax1.contourf(dilated_obstacle_mask, levels=1, colors='red', alpha=0.3)  # 添加膨胀障碍物掩码的显示
    ax1.set_title('A* Paths')
    ax1.legend()

    # 右边：贝塞尔曲线
    ax2.imshow(img)
    ax2.plot(bezier_curve_points[:, 0], bezier_curve_points[:, 1], 'r-', label='Bezier Curve')
    ax2.set_title('Bezier Curve')
    ax2.legend()

    # 计算并显示欧式距离
    original_distance = calculate_euclidean_distance(original_path)
    dilated_distance = calculate_euclidean_distance(dilated_path)
    bezier_distance = calculate_euclidean_distance(bezier_curve_points)

    # 在左图右下角显示原始 A* 和膨胀后 A* 的距离
    ax1.text(0.95, 0.05, f'Original Distance: {original_distance:.2f}\nDilated Distance: {dilated_distance:.2f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='blue')

    # 在右图右下角显示贝塞尔曲线的距离
    ax2.text(0.95, 0.05, f'Bezier Distance: {bezier_distance:.2f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='red')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

def calculate_curvature(bezier_curve_points):
    """
    计算贝塞尔曲线的曲率变化
    :param bezier_curve_points: 贝塞尔曲线点数组
    :return: 曲率数组
    """
    dx = np.gradient(bezier_curve_points[:, 0])
    dy = np.gradient(bezier_curve_points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def plot_curve_and_curvature(input_image_path, bezier_curve_points, output_filename):
    """
    绘制并保存贝塞尔曲线及其曲率变化的左右布局图
    :param input_image_path: 输入图像路径
    :param bezier_curve_points: 贝塞尔曲线点数组
    :param output_filename: 输出文件名
    """
    img = plt.imread(input_image_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制贝塞尔曲线
    ax1.imshow(img)
    ax1.plot(bezier_curve_points[:, 0], bezier_curve_points[:, 1], 'r-', label='Bezier Curve')
    ax1.set_title('Bezier Curve')
    ax1.legend()

    # 计算并绘制曲率变化
    curvature = calculate_curvature(bezier_curve_points)
    ax2.plot(curvature, 'b-')
    ax2.set_title('Curvature Variation')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Curvature')

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()