import numpy as np
import matplotlib.pyplot as plt

# 定义N的范围
N = np.linspace(500, 3000, 100)  # 从1到10的100个点

# 计算N和N log N的值
N_log_N = N * np.log(N)

# 创建图形
plt.figure()

# 绘制N的曲线
plt.plot(N, N, label='N', color='blue')

# 绘制N log N的曲线
plt.plot(N, N_log_N, label='N log N', color='orange')

# 添加图例
plt.legend()

# 添加标题和坐标轴标签
plt.title('Comparison of N and N log N')
plt.xlabel('N')
plt.ylabel('Value')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()


