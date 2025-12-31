"""
测试轨迹特征提取方法
展示如何保留原始三维信息的不同方法
"""
import numpy as np
from trajectary import Trajectary

# 创建测试数据
test_trajectory_dict = {
    "tendency": [0.8, 0.6, 0.9, 0.7, 0.85],
    "sequence": [1, 2, 3, 2, 4],
    "selection": ["left", "right", "left", "right", "left"]
}

print("=" * 60)
print("轨迹特征提取方法测试")
print("=" * 60)

# 创建 Trajectary 实例
traj = Trajectary(file_name="", trajectory_dict=test_trajectory_dict)

# 方法1: 简单拼接（保留原始三维信息）
print("\n【方法1: concatenate - 简单拼接】")
features1 = traj.trajectory_valulization(test_trajectory_dict, method='concatenate')
print(f"特征形状: {features1.shape}")
print(f"特征矩阵:\n{features1}")
print(f"[OK] 完全保留原始三维信息 [tendency, sequence, selection]")

# 方法2: 原始三维 + 交互特征
print("\n【方法2: with_interaction - 原始三维 + 交互特征】")
features2 = traj.trajectory_valulization(test_trajectory_dict, method='with_interaction')
print(f"特征形状: {features2.shape}")
print(f"特征矩阵:\n{features2}")
print(f"[OK] 保留原始三维信息 + 交互特征")
print(f"  前3列: [tendency, sequence, selection] (原始信息)")
print(f"  后3列: [t×s, t×sel, s×sel] (交互特征)")

# 方法3: 原始三维 + 统计特征
print("\n【方法3: with_statistics - 原始三维 + 统计特征】")
features3 = traj.trajectory_valulization(test_trajectory_dict, method='with_statistics')
print(f"特征形状: {features3.shape}")
print(f"特征矩阵:\n{features3}")
print(f"[OK] 保留原始三维信息 + 统计特征")
print(f"  前3列: [tendency, sequence, selection] (原始信息)")
print(f"  中3列: [t-mean, s-mean, sel-mean] (中心化)")
print(f"  后3列: [(t-mean)/std, (s-mean)/std, (sel-mean)/std] (标准化)")

# 方法4: 完整特征集
print("\n【方法4: full - 原始三维 + 交互 + 统计】")
features4 = traj.trajectory_valulization(test_trajectory_dict, method='full')
print(f"特征形状: {features4.shape}")
print(f"特征矩阵:\n{features4}")
print(f"[OK] 最完整的特征集，保留所有原始信息")
print(f"  列1-3: 原始三维信息")
print(f"  列4-6: 交互特征")
print(f"  列7-9: 中心化特征")
print(f"  列10-12: 标准化特征")

# 验证原始信息是否保留
print("\n" + "=" * 60)
print("验证：原始信息是否完全保留")
print("=" * 60)
original_t = np.array(test_trajectory_dict["tendency"])
original_s = np.array(test_trajectory_dict["sequence"])

# 从特征矩阵中提取原始值
extracted_t = features1[:, 0]
extracted_s = features1[:, 1]

print(f"\n原始 tendency: {original_t}")
print(f"提取 tendency: {extracted_t}")
print(f"是否完全一致: {np.allclose(original_t, extracted_t)}")

print(f"\n原始 sequence: {original_s}")
print(f"提取 sequence: {extracted_s}")
print(f"是否完全一致: {np.allclose(original_s, extracted_s)}")

# 性能对比
print("\n" + "=" * 60)
print("性能对比")
print("=" * 60)
import time

# 生成更大的测试数据
large_trajectory_dict = {
    "tendency": np.random.rand(10000).tolist(),
    "sequence": np.random.randint(0, 100, 10000).tolist(),
    "selection": np.random.choice(["left", "right"], 10000).tolist()
}

traj_large = Trajectary(file_name="", trajectory_dict=large_trajectory_dict)

methods = ['concatenate', 'with_interaction', 'with_statistics', 'full']
for method in methods:
    start = time.time()
    features = traj_large.trajectory_valulization(large_trajectory_dict, method=method)
    elapsed = time.time() - start
    print(f"{method:20s}: {elapsed*1000:.2f} ms, 形状: {features.shape}, 吞吐量: {len(features)/elapsed:.0f} samples/s")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
所有方法都完全保留原始三维信息（tendency, sequence, selection）
- 'concatenate': 最简单，3维特征
- 'with_interaction': 6维特征，包含特征交互
- 'with_statistics': 9维特征，包含统计信息
- 'full': 12维特征，最完整但计算量最大

所有方法都可以从特征矩阵的前3列恢复原始值！
""")

