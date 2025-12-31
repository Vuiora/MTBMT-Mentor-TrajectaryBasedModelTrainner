"""
测试轨迹标签功能
验证每条轨迹都有：检索时间、轨迹长度、决策效果的标签
"""
import numpy as np
from trajectary import Trajectary
import sys
import os

# 导入决策树轨迹生成器
try:
    from generate_decision_tree_trajectories import (
        generate_complex_decision_tree,
        DecisionTreeTrajectoryExtractor,
        convert_to_trajectary_format
    )
    HAS_DECISION_TREE_MODULE = True
except ImportError:
    HAS_DECISION_TREE_MODULE = False
    print("Warning: generate_decision_tree_trajectories module not found")

print("=" * 60)
print("轨迹标签功能测试")
print("=" * 60)

# 测试1: 完整标签数据
print("\n【测试1: 完整标签数据】")
test_trajectory_dict = {
    "tendency": [0.8, 0.6, 0.9, 0.7, 0.85],
    "sequence": [1, 2, 3, 2, 4],
    "selection": ["left", "right", "left", "right", "left"],
    "retrieval_time": 0.125,      # 检索时间（秒）
    "trajectory_length": 5,       # 轨迹长度（节点数）
    "decision_effect": 0.92       # 决策效果（如准确率）
}

traj1 = Trajectary(file_name="", trajectory_dict=test_trajectory_dict)
labels1 = traj1.get_trajectory_labels()

print(f"检索时间: {labels1['retrieval_time']} 秒")
print(f"轨迹长度: {labels1['trajectory_length']} 个节点")
print(f"决策效果: {labels1['decision_effect']}")
print("[OK] 所有标签都存在")

# 测试2: 缺少标签（自动补充）
print("\n【测试2: 缺少标签（自动补充）】")
test_trajectory_dict2 = {
    "tendency": [0.8, 0.6, 0.9],
    "sequence": [1, 2, 3],
    "selection": ["left", "right", "left"]
    # 缺少标签
}

traj2 = Trajectary(file_name="", trajectory_dict=test_trajectory_dict2)
labels2 = traj2.get_trajectory_labels()

print(f"检索时间: {labels2['retrieval_time']} (自动设置为默认值)")
print(f"轨迹长度: {labels2['trajectory_length']} (自动计算)")
print(f"决策效果: {labels2['decision_effect']} (自动设置为默认值)")
print("[OK] 缺少的标签已自动补充")

# 测试3: 手动设置标签
print("\n【测试3: 手动设置标签】")
test_trajectory_dict3 = {
    "tendency": [0.8, 0.6, 0.9, 0.7],
    "sequence": [1, 2, 3, 2],
    "selection": ["left", "right", "left", "right"]
}

traj3 = Trajectary(file_name="", trajectory_dict=test_trajectory_dict3)
traj3.set_trajectory_labels(
    retrieval_time=0.15,
    trajectory_length=4,
    decision_effect=0.88
)

labels3 = traj3.get_trajectory_labels()
print(f"检索时间: {labels3['retrieval_time']} 秒")
print(f"轨迹长度: {labels3['trajectory_length']} 个节点")
print(f"决策效果: {labels3['decision_effect']}")
print("[OK] 标签已手动设置")

# 测试4: 监督学习数据准备
print("\n【测试4: 监督学习数据准备】")
test_trajectory_dict4 = {
    "tendency": [0.8, 0.6, 0.9, 0.7, 0.85],
    "sequence": [1, 2, 3, 2, 4],
    "selection": ["left", "right", "left", "right", "left"],
    "retrieval_time": 0.125,
    "trajectory_length": 5,
    "decision_effect": 0.92
}

traj4 = Trajectary(file_name="", trajectory_dict=test_trajectory_dict4)
supervised_data = traj4.trajectory_supervised_learning()

print(f"特征矩阵形状: {supervised_data['features'].shape}")
print(f"特征矩阵前3行:\n{supervised_data['features'][:3]}")
print(f"标签: {supervised_data['labels']}")
print("[OK] 监督学习数据准备完成")

# 测试5: 完整处理流程
print("\n【测试5: 完整处理流程】")
test_trajectory_dict5 = {
    "tendency": [0.8, 0.6, 0.9, 0.7],
    "sequence": [1, 2, 3, 2],
    "selection": ["left", "right", "left", "right"],
    "retrieval_time": 0.12,
    "trajectory_length": 4,
    "decision_effect": 0.90
}

traj5 = Trajectary(file_name="", trajectory_dict=test_trajectory_dict5)
result = traj5.processing()

print(f"特征矩阵形状: {result['features'].shape}")
print(f"标签: {result['labels']}")
print("[OK] 完整处理流程成功")

# 测试6: 验证标签完整性
print("\n" + "=" * 60)
print("验证：所有轨迹都有完整标签")
print("=" * 60)

trajectories = [
    {"name": "轨迹1", "dict": test_trajectory_dict},
    {"name": "轨迹2", "dict": test_trajectory_dict2},
    {"name": "轨迹3", "dict": test_trajectory_dict3},
    {"name": "轨迹4", "dict": test_trajectory_dict4},
    {"name": "轨迹5", "dict": test_trajectory_dict5},
]

all_valid = True
for traj_info in trajectories:
    traj = Trajectary(file_name="", trajectory_dict=traj_info["dict"])
    labels = traj.get_trajectory_labels()
    
    has_all_labels = all(key in labels for key in ["retrieval_time", "trajectory_length", "decision_effect"])
    all_valid = all_valid and has_all_labels
    
    status = "[OK]" if has_all_labels else "[FAIL]"
    print(f"{status} {traj_info['name']}: "
          f"检索时间={labels['retrieval_time']}, "
          f"轨迹长度={labels['trajectory_length']}, "
          f"决策效果={labels['decision_effect']}")

print("\n" + "=" * 60)
if all_valid:
    print("总结: 所有轨迹都有完整的标签（检索时间、轨迹长度、决策效果）")
else:
    print("警告: 部分轨迹缺少标签")
print("=" * 60)

# 测试7: 从复杂决策树生成的轨迹数据
if HAS_DECISION_TREE_MODULE:
    print("\n" + "=" * 60)
    print("【测试7: 复杂决策树生成的轨迹数据】")
    print("=" * 60)
    
    try:
        # 生成多个决策树并提取轨迹（使用验证集，避免数据泄露）
        print("\n生成多个决策树（必要的和不必要的）...")
        print("[数据安全] 使用三部分数据划分：训练集/验证集/测试集")
        print("[决策效果] 每个决策树的决策成功率在测试集上评估")
        
        from generate_decision_tree_trajectories import generate_multiple_decision_trees
        
        # 生成多个决策树
        trees, X_train, X_val, X_test, y_train, y_val, y_test = generate_multiple_decision_trees(
            use_validation_set=True, n_trees=5  # 生成5个决策树用于测试
        )
        
        # 使用验证集提取轨迹（避免测试集数据泄露）
        n_test_samples = 20  # 每个决策树提取20个样本
        print(f"\n为每个决策树提取 {n_test_samples} 个验证集样本的轨迹...")
        print("[数据安全] 使用验证集提取轨迹，测试集保留用于最终评估")
        
        all_trajectories = []
        for tree_idx, (clf, decision_success_rate, tree_info) in enumerate(trees):
            print(f"\n处理决策树 {tree_idx+1}: {tree_info['name']} (成功率: {decision_success_rate:.4f})")
            
            extractor = DecisionTreeTrajectoryExtractor(clf)
            trajectories = extractor.extract_trajectories_batch(
                X_val[:n_test_samples],  # 使用验证集
                y_val[:n_test_samples]
            )
            
            # 转换为trajectary格式，使用该决策树的成功率作为决策效果
            formatted_trajectories = convert_to_trajectary_format(
                trajectories,
                retrieval_time=0.001,
                decision_effect=decision_success_rate  # 使用该决策树的测试准确率
            )
            
            # 添加决策树信息
            for traj in formatted_trajectories:
                traj["tree_name"] = tree_info["name"]
                traj["tree_idx"] = tree_idx
            
            all_trajectories.extend(formatted_trajectories)
        
        formatted_trajectories = all_trajectories
        
        print(f"\n成功生成 {len(formatted_trajectories)} 条轨迹")
        
        # 测试每条轨迹
        all_valid_dt = True
        for i, traj_dict in enumerate(formatted_trajectories[:1000]):  # 测试前1000条
            print(f"\n--- 轨迹 {i+1} ---")
            traj = Trajectary(file_name="", trajectory_dict=traj_dict)
            labels = traj.get_trajectory_labels()
            
            has_all_labels = all(key in labels for key in ["retrieval_time", "trajectory_length", "decision_effect"])
            all_valid_dt = all_valid_dt and has_all_labels
            
            status = "[OK]" if has_all_labels else "[FAIL]"
            print(f"{status} 检索时间={labels['retrieval_time']}, "
                  f"轨迹长度={labels['trajectory_length']}, "
                  f"决策效果={labels['decision_effect']:.4f}")
            
            # 测试特征提取
            try:
                features = traj.trajectory_valulization(method='full')
                print(f"  特征矩阵形状: {features.shape}")
                
                # 测试监督学习数据准备
                supervised_data = traj.trajectory_supervised_learning()
                print(f"  监督学习特征形状: {supervised_data['features'].shape}")
                print(f"  监督学习标签: {supervised_data['labels']}")
            except Exception as e:
                print(f"  [ERROR] 特征提取失败: {e}")
                all_valid_dt = False
        
        # 统计信息
        print("\n轨迹统计信息:")
        lengths = [t["trajectory_length"] for t in formatted_trajectories]
        print(f"  平均轨迹长度: {np.mean(lengths):.2f} 节点")
        print(f"  最短轨迹: {np.min(lengths)} 节点")
        print(f"  最长轨迹: {np.max(lengths)} 节点")
        
        # 测试批量处理
        print("\n测试批量处理多条轨迹...")
        batch_results = []
        for traj_dict in formatted_trajectories[:1000]:
            traj = Trajectary(file_name="", trajectory_dict=traj_dict)
            result = traj.processing()
            batch_results.append(result)
        
        print(f"成功处理 {len(batch_results)} 条轨迹")
        print(f"所有轨迹特征维度一致: {all(r['features'].shape[1] == 12 for r in batch_results)}")
        
        print("\n" + "=" * 60)
        if all_valid_dt:
            print("[OK] 所有从决策树生成的轨迹都通过了测试")
        else:
            print("[WARNING] 部分轨迹测试失败")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] 决策树轨迹测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n跳过测试7: 决策树模块未找到")
    print("请先运行: python generate_decision_tree_trajectories.py")

