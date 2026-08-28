import contextlib

import torch
import pickle
import numpy as np
from datasets import Dataset
from torch import Tensor, nn
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import LedoitWolf
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_X_y, check_is_fitted

from .hidden_state import get_hidden_states


def get_steering_vector(
    prompts_a: Dataset,
    prompts_b: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_key: str = "formatted_prompt",
    token_idx: int = -1,
    layer_ratio: float = 2 / 3,
) -> torch.Tensor:
    hidden_states_a, _ = get_hidden_states(
        model,
        tokenizer,
        prompts_a,
        prompt_key=prompt_key,
        max_new_tokens=1,
    )
    hidden_states_b, _ = get_hidden_states(
        model,
        tokenizer,
        prompts_b,
        prompt_key=prompt_key,
        max_new_tokens=1,
    )

    n_layers = len(hidden_states_a)
    target_layer = int(n_layers * layer_ratio)
    va = hidden_states_a[target_layer].mean(dim=0)[token_idx]
    vb = hidden_states_b[target_layer].mean(dim=0)[token_idx]
    v = vb - va
    return v


class Steerer(contextlib.AbstractContextManager):
    def __init__(
        self,
        vector: torch.Tensor,
        model: nn.Module,
        layers: list[int],
        layer_types: list[str],
        strength: float = 1.0,
        mode: str = "absolute",
        steering_func: Callable = None,
        layer_template: str = "model.layers.{layer_index}.{layer_type}",
        enabled: bool = True,
        token_mask: torch.Tensor | None = None,
    ):
        self.model = model
        self.vector = vector
        self.layers = layers
        self.layer_types = layer_types
        self.layer_template = layer_template
        self.strength = strength
        self.mode = mode
        self.enabled = enabled
        self.token_mask = token_mask
        self.steering_func = steering_func
        self._hooks = []

    def _make_hook(self, steering_vector, strength):
        def hook(module, inputs, output):
            nonlocal steering_vector

            if isinstance(output, Tensor):
                hidden_states = output
            elif isinstance(output, tuple):
                hidden_states = output[0]
            else:
                raise NotImplementedError(f"Unsupported output type: {type(output)}")

            steering_vector = steering_vector.to(hidden_states.device).float()
            original_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()

            if self.mode == "relative":
                scalar = torch.abs((hidden_states @ steering_vector).unsqueeze(-1))
                v = scalar * steering_vector
            elif self.mode == "absolute":
                v = steering_vector
            else:
                raise NotImplementedError(f"Invalid mode: {self.mode}")
            
            delta = v * strength
            modified_hidden_states = (hidden_states + delta).to(original_dtype)
            
            if isinstance(output, tuple):
                return (modified_hidden_states, *output[1:])
            return modified_hidden_states

        return hook

    def __enter__(self):
        if not self.enabled:
            return self

        for layer_index in self.layers:
            for layer_type in self.layer_types:
                layer_name = self.layer_template.format(
                    layer_index=layer_index, layer_type=layer_type
                )
                module = self.model.get_submodule(layer_name)

                hook_handle = module.register_forward_hook(
                    self._make_hook(self.vector, self.strength)
                )
                self._hooks.append(hook_handle)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return False

        # Remove all registered hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False



class CalibratedAsymmetricLDA(BaseEstimator, ClassifierMixin):
    """
    Calibrated Asymmetric LDA
    
    主要特性:
    1. 单边白化 (One-Sided Whitening): 仅利用指令集 (Concentrated) 的协方差计算方向。
    2. FPR 阈值校准 (FPR Calibration): 基于良性数据的分布设定截距，而非两类中点。
    3. 自适应 Steering: 基于良性数据的噪声水平动态调整干预强度。
    """
    def __init__(self, concentrated_label=1, target_fpr=0.01):
        self.concentrated_label = concentrated_label
        self.target_fpr = target_fpr  # 目标误报率 (e.g. 1%)
        
    def fit(self, X, y):
        """
        训练模型。
        X: (N_samples, N_features)
        y: (N_samples,) 标签
        """
        # 1. 检查输入
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # 2. 数据分离
        X_conc = X[y == self.concentrated_label]       # 指令/恶意 (Instruction)
        X_other = X[y != self.concentrated_label]      # 良性/背景 (Knowledge)
        
        if X_conc.shape[0] < 2:
            raise ValueError(f"Concentrated class {self.concentrated_label} samples too few.")

        # 3. 计算均值
        self.mean_conc_ = np.mean(X_conc, axis=0)
        self.mean_other_ = np.mean(X_other, axis=0)

        # 4. 核心步骤：仅在指令集上计算 Precision Matrix (单边白化)
        # 使用 Ledoit-Wolf 估计以处理高维奇异性
        cov_estimator = LedoitWolf(store_precision=True)
        cov_estimator.fit(X_conc)
        self.precision_ = cov_estimator.precision_ 

        # 5. 计算投影方向 w (Fisher Variant)
        # w = Sigma_I^{-1} * (mu_I - mu_K)
        diff_mean = self.mean_conc_ - self.mean_other_
        self.coef_ = self.precision_ @ diff_mean
        
        # 归一化 w，方便后续计算距离
        self.coef_ = self.coef_ / np.linalg.norm(self.coef_)

        # 6. 计算截距 (FPR Calibration)
        # 计算所有良性数据在方向 w 上的原始投影值 (不加截距)
        benign_projections = X_other @ self.coef_
        
        # 我们希望: Score = proj + intercept < 0 对 (1-fpr) 的良性数据成立
        # 即: intercept < -proj
        cutoff_quantile = 1.0 - self.target_fpr
        boundary_val = np.quantile(benign_projections, cutoff_quantile)
        
        self.intercept_ = -boundary_val
        
        # 7. 保存统计量用于 Steering
        # 计算良性数据在投影方向上的标准差 (噪声水平)
        self.benign_std_projected_ = np.std(benign_projections)

        return self

    def decision_function(self, X):
        """计算带符号的距离 (Score)"""
        check_is_fitted(self, ['coef_', 'intercept_'])
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """预测类别 (Score > 0 为 Concentrated Class)"""
        scores = self.decision_function(X)
        other_label = self.classes_[self.classes_ != self.concentrated_label][0]
        return np.where(scores > 0, self.concentrated_label, other_label)

    def get_steering_vector(self):
        """获取单位 Steering 向量"""
        return self.coef_

    def adaptive_steer(self, hidden_states, margin_sigma=3.0, aggressive=False):
        """
        对 PyTorch Tensor 进行自适应 Steering。
        
        Args:
            hidden_states: (Batch, Seq, Dim) or (Batch, Dim) PyTorch Tensor
            margin_sigma: 安全边界系数。我们要把攻击推回 0 线以下多少个 sigma 处？
            aggressive: 如果 True, 对所有 Score > 0 的样本进行干预；
                        如果 False, 仅当预测为 target 且 score > 0 时干预。
        """
        is_batch_seq = hidden_states.dim() == 3
        if is_batch_seq:
            # 暂时展平处理，或者只处理最后一个 token (根据实际需求调整)
            # 这里演示处理 batch 中的每个向量
            original_shape = hidden_states.shape
            X_tensor = hidden_states.reshape(-1, original_shape[-1])
        else:
            X_tensor = hidden_states

        device = X_tensor.device
        dtype = X_tensor.dtype
        
        # 转为 numpy 计算 score
        X_np = X_tensor.detach().float().cpu().numpy()
        scores = self.decision_function(X_np)
        
        # 设定安全边界: 我们希望将攻击推回到 Score = - (sigma * benign_std)
        safe_margin = margin_sigma * self.benign_std_projected_
        target_score = -safe_margin

        # 计算 Diff: 当前分数与目标分数的差距
        # 我们只关心 scores > target_score 的部分 (即攻击样本，或过于接近边界的样本)
        # diff = current - target
        diffs = scores - target_score
        diffs = np.maximum(diffs, 0) # 如果已经在安全区深处 (diff < 0)，则不干预

        # 计算 Shift 向量
        # Shift = - (diff) * w_unit



class MultiLayerVotingDetector:
    def __init__(self, layers, threshold=0.5):
        """
        Args:
            layers: 需要检测的层索引列表，例如 [10, 15, 20, 25]
        """
        self.layers = layers
        self.threshold = threshold
        self.detectors = {}  # {layer_id: CalibratedAsymmetricLDA}

    def fit(self, hidden_states_dict, y, target_fpr_per_layer=0.05):
        """
        Args:
            hidden_states_dict: 字典 {layer_id: numpy array (N, dim)}
                                包含每一层的训练数据
            y: 标签 (N,)
            target_fpr_per_layer: 为了保证总体的 Recall，单层可以稍微松一点
        """
        print(f"开始训练 {len(self.layers)} 个层的检测器...")
        
        for layer_id in self.layers:
            if layer_id not in hidden_states_dict:
                raise ValueError(f"缺少层 {layer_id} 的训练数据")
            
            X_layer = hidden_states_dict[layer_id]
            
            # 初始化并训练单层检测器 (使用 CalibratedAsymmetricLDA)
            detector = CalibratedAsymmetricLDA(
                concentrated_label=1, 
                target_fpr=target_fpr_per_layer
            )
            detector.fit(X_layer, y)
            detector.layer_id = layer_id  # 保存 layer_id 用于后续引用
            self.detectors[layer_id] = detector
            
        print("多层检测器训练完成。")
        return self



    def predict(self, hidden_states_dict, soft_threshold=0.0, hard_threshold=None):
        """
        同时返回 soft 和 hard 两种投票模式的预测结果，避免重复前向推理
        
        Args:
            hidden_states_dict: 字典 {layer_id: numpy array (N, dim)}
            soft_threshold: soft 模式的阈值 (默认 0.0)
            hard_threshold: hard 模式的阈值 (默认为层数的一半+1)
        
        Returns:
            results: dict 包含以下键值:
                - 'stack_scores': (n_layers, N) 每层的分数
                - 'stack_decisions': (n_layers, N) 每层的决策
                - 'soft_aggregate_score': (N,) soft 模式聚合分数 (平均)
                - 'soft_pred': (N,) soft 模式预测
                - 'hard_votes': (N,) hard 模式投票数
                - 'hard_pred': (N,) hard 模式预测
        """
        layer_scores = []
        layer_decisions = []
        
        # 1. 收集每一层的结果 (只需要一次前向传播)
        for layer_id in self.layers:
            X_layer = hidden_states_dict[layer_id]
            detector = self.detectors[layer_id]
            
            score = detector.decision_function(X_layer)  # (N,)
            decision = (score > 0).astype(int)
            
            layer_scores.append(score)
            layer_decisions.append(decision)
            
        # Stack 起来: (n_layers, N)
        stack_scores = np.vstack(layer_scores)
        stack_decisions = np.vstack(layer_decisions)
        
        # 2. Soft 模式: 平均分数
        soft_aggregate_score = np.mean(stack_scores, axis=0)
        soft_pred = (soft_aggregate_score > soft_threshold).astype(int)
        
        # 3. Hard 模式: 投票计数
        hard_votes = np.sum(stack_decisions, axis=0)
        if hard_threshold is None:
            hard_threshold = len(self.layers) // 2 + 1
        hard_pred = (hard_votes >= hard_threshold).astype(int)
        
        return {
            'stack_scores': stack_scores,
            'stack_decisions': stack_decisions,
            'soft_aggregate_score': soft_aggregate_score,
            'soft_pred': soft_pred,
            'hard_votes': hard_votes,
            'hard_pred': hard_pred,
            'soft_threshold': soft_threshold,
            'hard_threshold': hard_threshold,
        }

    def detect_and_steer(self, hidden_states_dict, steer_strength=3.0):
        """
        多层联动防御逻辑：
        只有当【聚合结果】判定为攻击时，才触发 Steering。
        一旦触发，对【所有检测到攻击信号的层】进行 Steering。
        
        Returns:
            steered_states_dict: 修改后的隐藏状态
            is_attack: 是否触发了防御
        """
        is_attack_batch, agg_scores, stack_scores, _ = self.predict_detailed(hidden_states_dict)
        
        # 注意：这里演示的是 batch size = 1 的情况，或者简单处理
        # 如果是 batch 处理，需要对 batch 里的每个样本分别判断
        
        steered_dict = {}
        
        # 遍历每一层
        for i, layer_id in enumerate(self.layers):
            original_tensor = hidden_states_dict[layer_id] # 假设是 tensor
            detector = self.detectors[layer_id]
            
            # 判断是否需要 steer 这个层
            # 策略 A: 只要总结果是攻击，所有层都修补 (保守)
            # 策略 B: 总结果是攻击，且该层分数 > 0 才修补 (精准) -> 推荐
            
            # 将 Tensor 转 numpy 供 detector 计算
            X_np = original_tensor.detach().cpu().numpy()
            layer_score = detector.decision_function(X_np)
            
            # 构建 mask: 总体认为是攻击 AND 当前层认为是攻击
            # (处理 Batch 维度)
            mask = (is_attack_batch == 1) & (layer_score > 0)
            
            if np.any(mask):
                # 调用单层的 adaptive steer 逻辑
                # 这里我们需要稍微修改一下 adaptive_steer 接口以支持 mask
                # 或者简单的循环：
                
                # 计算 Steering 向量
                # Shift = (Score + margin) * -w
                margin = steer_strength * detector.score_std_
                diff = layer_score - (-margin)
                diff = np.maximum(diff, 0)
                
                # 应用 mask
                diff = diff * mask.astype(float)
                
                w_tensor = torch.tensor(detector.coef_, device=original_tensor.device, dtype=original_tensor.dtype)
                shift = -(torch.tensor(diff, device=original_tensor.device).unsqueeze(-1)) * w_tensor
                
                steered_dict[layer_id] = original_tensor + shift
            else:
                steered_dict[layer_id] = original_tensor
                
        return steered_dict, is_attack_batch



class AsymmetricLDA(BaseEstimator, ClassifierMixin):
    """
    Asymmetric LDA
    """
    def __init__(self, concentrated_label=1, layer_id=None, token_idx=None):
        self.concentrated_label = concentrated_label
        self.layer_id = layer_id
        self.token_idx = token_idx

    def fit(self, X, y):
        # 1. check the input data
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # confirm only two classes
        if len(self.classes_) != 2:
            raise ValueError("AsymmetricLDA currently only supports binary classification problems.")

        # 2. separate the data
        # the concentrated class (Concentrated)
        X_conc = X[y == self.concentrated_label]
        # the dispersed class (Dispersed / Other)
        X_other = X[y != self.concentrated_label]

        if X_conc.shape[0] < 2:
            raise ValueError(f"The number of samples for class {self.concentrated_label} is too small, cannot calculate the covariance.")

        # 3. calculate the mean
        self.mean_conc_ = np.mean(X_conc, axis=0)
        self.mean_other_ = np.mean(X_other, axis=0)

        # 4. core: estimate the covariance only on the concentrated class (using LedoitWolf to handle high-dimensional/singular problems)
        # store_precision=True will directly calculate the inverse matrix (Precision Matrix)
        cov_estimator = LedoitWolf(store_precision=True)
        cov_estimator.fit(X_conc)

        self.precision_ = cov_estimator.precision_ # S^-1

        # 5. calculate the projection direction w (Fisher's criterion variant)
        # w = S_conc^-1 * (mu_conc - mu_other)
        diff_mean = self.mean_conc_ - self.mean_other_
        self.coef_ = self.precision_ @ diff_mean

        # 6. calculate the intercept (Bias)
        # to move the boundary (Score=0) to the middle of the projected means of the two classes, we need to translate
        # Intercept = -0.5 * w.T * (mu_conc + mu_other)
        mid_point = 0.5 * (self.mean_conc_ + self.mean_other_)
        self.intercept_ = -np.dot(self.coef_, mid_point)

        return self
    
    def save(self, file_path):
        """save the model"""
        params = {
            'concentrated_label': self.concentrated_label,
            'layer_id': self.layer_id,
            'token_idx': self.token_idx,
            'coef_': self.coef_,
            'intercept_': self.intercept_,
        }
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
    
    def load(self, file_path):
        """load the model"""
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        self.concentrated_label = params['concentrated_label']
        self.layer_id = params['layer_id']
        self.token_idx = params['token_idx']
        self.coef_ = params['coef_']
        self.intercept_ = params['intercept_']

    def decision_function(self, X):
        """return the signed distance to the boundary (Score)"""
        check_is_fitted(self, ['coef_', 'intercept_'])
        # Linear projection: z = w.T * x + b
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        """predict the class based on the score > 0"""
        scores = self.decision_function(X)
        # if score > 0, predict the concentrated_label, otherwise predict the other class
        predictions = np.where(scores > 0, self.concentrated_label, self.classes_[self.classes_!=self.concentrated_label][0])
        return predictions

    def steer(self, X, coefficient=1.0):
        """
        coefficient > 0: enhance the concept
        coefficient < 0: suppress the concept
        """
        w_norm = self.coef_ / np.linalg.norm(self.coef_)
        return X - coefficient * w_norm

    def unit_steer_direction(self):
        """get the steering direction"""
        return self.coef_ / np.linalg.norm(self.coef_)
    
    def adaptive_steer_torch(self, hidden_states, target_class=0, margin=1.0):
        """adaptive steering: only modify the last token"""
        shifts_np = self.adaptive_steer(
            hidden_states.detach().cpu().numpy(), 
            target_class=target_class, 
            margin=margin
        )[1]
        shifts_tensor = torch.from_numpy(shifts_np).to(
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        return shifts_tensor

    def adaptive_steer(self, X, target_class=0, margin=1.0):
        """
        adaptive steering:
        only modify the data that are 'wrongly classified', and the modification amount 
        is the minimum distance required to cross the boundary.
        
        Parameters
        ----------
        X : input data (N, 4096)
        target_class : which class do you want the data to be? (0 or 1)
        margin : how far to go inside the boundary after crossing the boundary? (safe distance)

        Returns
        -------
        X_steered : modified data
        """
        # 1. obtain the steering direction
        w_norm = np.linalg.norm(self.coef_)
        w_unit = self.unit_steer_direction()

        # 2. calculate the current score of the data
        # score > 0 denotes Class 1 (Concentrated), score < 0 denotes Class 0
        current_scores = self.decision_function(X)

        if target_class == 0:
            # target score is -margin
            # we only modify the data where score > -margin (i.e. Class 1 data, or data too close to the boundary)
            # diff = current score - target score
            diffs = current_scores - (-margin)
            # if diff < 0, the data is already in the target region, no need to modify, set to 0
            diffs = np.maximum(diffs, 0)

            # the direction is the negative gradient direction (-w)
            # the shift vector = (diff / ||w||) * (-w_unit)
            # because w_unit is already w/||w||, score is w*x, so the coefficient is directly diff / ||w||
            shifts = -(diffs[:, np.newaxis] / w_norm) * w_unit

        else: # target_class == 1
            # target score is +margin
            # we only modify the data where score < margin (i.e. Class 0 data, or data too close to the boundary)
            diffs = margin - current_scores
            diffs = np.maximum(diffs, 0)

            # the direction is the positive gradient direction (+w)
            shifts = (diffs[:, np.newaxis] / w_norm) * w_unit

        return X + shifts, shifts

    def transform(self, X):
        """transform the data to the projected space"""
        return self.decision_function(X).reshape(-1, 1)



def condition_similarity(direction, hidden_state):
    condition_projector = torch.ger(direction, direction) / torch.dot(direction, direction)
    projected_hidden_state = torch.tanh(torch.matmul(hidden_state, condition_projector.T))
    condition_sim = torch.cosine_similarity(hidden_state, projected_hidden_state).item()
    return condition_sim



def obtain_direction(
    hs_layer_token, 
    label, 
    layer_ids,
    use_pca = True,
    direction_mode: str = "pca_center",
    normalize: bool = False
):
    directions: dict[int, np.ndarray] = {}
    explained_variances: dict[int, float] = {}

    for layer_id in layer_ids:
        hs_layer_token[layer_id] = torch.from_numpy(hs_layer_token[layer_id])
        hs_label_0 = hs_layer_token[layer_id][label == 0]
        hs_label_1 = hs_layer_token[layer_id][label == 1]
        mean_hs_label_0 = hs_label_0.mean(dim=0, keepdim=True)
        mean_hs_label_1 = hs_label_1.mean(dim=0, keepdim=True)
        if direction_mode == "pca_diff":
            direction = mean_hs_label_1 - mean_hs_label_0
        elif direction_mode == "pca_center":
            center = hs_layer_token[layer_id].mean(axis=0)
            direction = hs_layer_token[layer_id] - center
        elif direction_mode == "pca_pairwise":
            hs_label_0 = hs_layer_token[layer_id][label == 0]
            hs_label_1 = hs_layer_token[layer_id][label == 1]
            center = (hs_label_0 + hs_label_1) / 2
            direction = hs_layer_token[layer_id] - center.expand_as(hs_layer_token[layer_id])
        else:
            raise ValueError(f"Invalid direction_mode: {direction_mode}")
        
        if not use_pca:
            directions[layer_id] = direction
            explained_variances[layer_id] = direction
            continue

        pca_model = PCA(n_components=1, whiten=False).fit(direction)
        directions[layer_id] = pca_model.components_.astype(np.float32).squeeze(axis=0)
        explained_variances[layer_id] = pca_model.explained_variance_ratio_[0]
    
        if normalize:
            directions[layer_id] = directions[layer_id] / directions[layer_id].norm(dim=1, keepdim=True)
    
    return directions, explained_variances