// CART决策树实现（回归树和分类树简单实现，重点实现二元切分的思想）
use std::collections::{HashSet, HashMap};

#[derive(Debug)]
pub enum Node {
    Leaf { value: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    }
}

/// 判断y是否可以作为分类树处理（如果全为整数且种类有限），否则回归树
fn is_classification(y: &Vec<f64>) -> bool {
    let mut set = HashSet::new();
    for &v in y {
        if v != v.floor() || v.is_nan() {
            return false;
        }
        set.insert(v as i64);
        if set.len() > 20 { // 超过20类视为回归
            return false;
        }
    }
    true
}

/// 多数投票取众数
fn majority(y: &[f64]) -> f64 {
    let mut count = HashMap::new();
    for &val in y {
        let key = val as i64; // 分类标签是整数，转换为i64作为键
        *count.entry(key).or_insert(0) += 1;
    }
    count.into_iter().max_by_key(|&(_, v)| v).unwrap().0 as f64
}

/// 计算分割后的不纯度(分类：基尼指数/信息熵，回归：方差加权)
fn split_score(y_left: &[f64], y_right: &[f64], classification: bool) -> f64 {
    let n = (y_left.len() + y_right.len()) as f64;
    if classification {
        let gini = |ys: &[f64]| {
            let mut counts = HashMap::new();
            for &v in ys {
                let key = v as i64; // 分类标签是整数，转换为i64作为键
                *counts.entry(key).or_insert(0) += 1;
            }
            let m = ys.len() as f64;
            1.0 - counts.values().map(|&c| (c as f64 / m).powi(2)).sum::<f64>()
        };
        (y_left.len() as f64 * gini(y_left) + y_right.len() as f64 * gini(y_right)) / n
    } else {
        let var = |ys: &[f64]| {
            let m = ys.len() as f64;
            if m == 0.0 { return 0.0; }
            let mean = ys.iter().sum::<f64>() / m;
            ys.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / m
        };
        (y_left.len() as f64 * var(y_left) + y_right.len() as f64 * var(y_right)) / n
    }
}

/// 训练决策树，X: n_samples x n_features, y: n_samples
pub fn cart_train(
    x: &Vec<Vec<f64>>,
    y: &Vec<f64>,
    max_depth: usize,
    min_samples_split: usize,
) -> Node {
    let classification = is_classification(y);
    cart_split(x, y, max_depth, min_samples_split, classification)
}

fn cart_split(
    x: &Vec<Vec<f64>>,
    y: &Vec<f64>,
    max_depth: usize,
    min_samples_split: usize,
    classification: bool,
) -> Node {
    if max_depth == 0 || y.len() < min_samples_split {
        if classification {
            return Node::Leaf { value: majority(y) };
        } else {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            return Node::Leaf { value: mean };
        }
    }
    let n_features = x[0].len();
    let mut best_feature = None;
    let mut best_thresh = 0.0;
    let mut best_score = f64::INFINITY;
    let mut best_left_idx = vec![];
    let mut best_right_idx = vec![];

    for feature in 0..n_features {
        // 取所有feature值并排序
        let mut xt: Vec<(f64, usize)> = x.iter().enumerate().map(|(i, xi)| (xi[feature], i)).collect();
        xt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for i in 1..xt.len() {
            let (thresh_l, _idx_l) = xt[i-1];
            let (thresh_r, _idx_r) = xt[i];
            if thresh_l == thresh_r { continue; }
            let threshold = (thresh_l + thresh_r) / 2.0;
            let mut left_idx = vec![];
            let mut right_idx = vec![];
            for (v, idx) in &xt {
                if *v <= threshold {
                    left_idx.push(*idx);
                } else {
                    right_idx.push(*idx);
                }
            }
            if left_idx.len() < min_samples_split || right_idx.len() < min_samples_split {
                continue;
            }
            let y_left = left_idx.iter().map(|&i| y[i]).collect::<Vec<_>>();
            let y_right = right_idx.iter().map(|&i| y[i]).collect::<Vec<_>>();
            let score = split_score(&y_left, &y_right, classification);
            if score < best_score {
                best_score = score;
                best_feature = Some(feature);
                best_thresh = threshold;
                best_left_idx = left_idx;
                best_right_idx = right_idx;
            }
        }
    }

    if let Some(feat) = best_feature {
        let x_left = best_left_idx.iter().map(|&i| x[i].clone()).collect::<Vec<_>>();
        let y_left = best_left_idx.iter().map(|&i| y[i]).collect::<Vec<_>>();
        let x_right = best_right_idx.iter().map(|&i| x[i].clone()).collect::<Vec<_>>();
        let y_right = best_right_idx.iter().map(|&i| y[i]).collect::<Vec<_>>();
        let left = cart_split(&x_left, &y_left, max_depth - 1, min_samples_split, classification);
        let right = cart_split(&x_right, &y_right, max_depth - 1, min_samples_split, classification);
        Node::Split {
            feature: feat,
            threshold: best_thresh,
            left: Box::new(left),
            right: Box::new(right),
        }
    } else {
        // 不能再分, 返回叶节点
        if classification {
            Node::Leaf { value: majority(y) }
        } else {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            Node::Leaf { value: mean }
        }
    }
}

/// 预测单个样本
pub fn cart_predict(tree: &Node, sample: &Vec<f64>) -> f64 {
    match tree {
        Node::Leaf { value } => *value,
        Node::Split { feature, threshold, left, right } => {
            if sample[*feature] <= *threshold {
                cart_predict(&left, sample)
            } else {
                cart_predict(&right, sample)
            }
        }
    }
}

/// 预测多个样本
pub fn cart_predict_batch(tree: &Node, x: &Vec<Vec<f64>>) -> Vec<f64> {
    x.iter().map(|xi| cart_predict(tree, xi)).collect()
}

