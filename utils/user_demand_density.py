import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import gaussian_kde
import matplotlib.patches as patches

# ------------------------------
# 1. 用户随机位置 & 需求
# ------------------------------
#np.random.seed(42)
num_users = 100
x = np.random.uniform(0, 5, num_users)
y = np.random.uniform(0, 5, num_users)
r = np.random.uniform(20, 40, num_users)  # 每个用户的需求

# ------------------------------
# 2. KDE 生成需求分布
# ------------------------------
values = np.vstack([x, y])
kde = gaussian_kde(values, weights=r, bw_method=0.3)

X, Y = np.meshgrid(np.linspace(0, 5, 200), np.linspace(0, 5, 200))
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

# ------------------------------
# 3. UAV 位置
# ------------------------------
uav_positions = np.array([[1, 3], [1.2, 1], [3, 1.5], [3.2, 3.8]])

# ------------------------------
# 4. 绘制热力图 + 用户 + UAV
# ------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# 热力图
contour = ax.contourf(X, Y, Z, levels=50, cmap="coolwarm")
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label("Downlink Demand (Mb)")

# 用户位置（画矩形，模仿手机图标）
for xi, yi in zip(x, y):
    rect = patches.Rectangle((xi-0.05, yi-0.05), 0.1, 0.1,
                             linewidth=0.5, edgecolor="black", facecolor="none", alpha=0.6)
    ax.add_patch(rect)

# UAV 图标（需要一张无人机图 png，比如 drone.png）
uav_icon = plt.imread("drone.png")  # 你需要准备一张无人机图标
for i, (ux, uy) in enumerate(uav_positions):
    imagebox = OffsetImage(uav_icon, zoom=0.025)  # 调整缩放比例
    ab = AnnotationBbox(imagebox, (ux, uy), frameon=False)
    ax.add_artist(ab)
    ax.text(ux+0.1, uy+0.1, f"UAV {i}", fontsize=10, color="black", weight="bold")

# 坐标设置
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_title("UAV Deployment and User Demand Heatmap")

plt.show()
