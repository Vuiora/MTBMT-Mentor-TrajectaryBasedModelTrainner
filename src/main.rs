use cart_decision_tree::{cart_train, cart_predict, cart_predict_batch};

fn main() {
    println!("=== CART决策树测试 ===\n");

    // 测试1: 分类树测试
    println!("【测试1: 分类树】");
    let x_class = vec![
        vec![2.0, 3.0],
        vec![1.0, 2.0],
        vec![3.0, 1.0],
        vec![4.0, 3.0],
        vec![2.0, 1.0],
        vec![5.0, 2.0],
    ];
    let y_class = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    
    println!("训练数据:");
    for (i, (xi, yi)) in x_class.iter().zip(y_class.iter()).enumerate() {
        println!("  样本{}: X={:?}, y={}", i, xi, yi);
    }
    
    let tree_class = cart_train(&x_class, &y_class, 3, 2);
    println!("\n训练后的决策树: {:?}", tree_class);
    
    let test_sample = vec![2.5, 2.0];
    let prediction = cart_predict(&tree_class, &test_sample);
    println!("预测样本 {:?} 的类别: {}", test_sample, prediction);
    
    // 测试2: 回归树测试
    println!("\n【测试2: 回归树】");
    let x_reg = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
        vec![6.0],
    ];
    let y_reg = vec![2.1, 4.2, 6.1, 8.0, 10.1, 12.2];
    
    println!("训练数据:");
    for (i, (xi, yi)) in x_reg.iter().zip(y_reg.iter()).enumerate() {
        println!("  样本{}: X={:?}, y={}", i, xi, yi);
    }
    
    let tree_reg = cart_train(&x_reg, &y_reg, 3, 2);
    println!("\n训练后的决策树: {:?}", tree_reg);
    
    let test_samples = vec![vec![2.5], vec![4.5], vec![7.0]];
    let predictions = cart_predict_batch(&tree_reg, &test_samples);
    println!("预测结果:");
    for (xi, pred) in test_samples.iter().zip(predictions.iter()) {
        println!("  样本 {:?} 的预测值: {:.2}", xi, pred);
    }
    
    // 测试3: 多特征分类
    println!("\n【测试3: 多特征分类】");
    let x_multi = vec![
        vec![1.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ];
    let y_multi = vec![1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    
    let tree_multi = cart_train(&x_multi, &y_multi, 4, 1);
    println!("训练后的决策树: {:?}", tree_multi);
    
    let test_multi = vec![vec![0.5, 0.5, 0.5]];
    let pred_multi = cart_predict(&tree_multi, &test_multi[0]);
    println!("预测样本 {:?} 的类别: {}", test_multi[0], pred_multi);
    
    println!("\n=== 测试完成 ===");
}

