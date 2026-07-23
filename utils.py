import json
import torch
import dataclasses
from pathlib import Path
from typing import Dict, Any, Optional

from models import OpticalConv2d, OpticalConvTranspose2d
import torch.nn as nn

class RunManager:
    def __init__(
            self,
            base_dir: str = "runs",
            exp_name: str = "cm_train",
            description: str = "",
            monitor_metric: str = "loss",  # 你想要追踪的最优指标名称
            mode: str = "min"  # "min" 表示越小越好，"max" 表示越大越好
    ):
        self.base_dir = Path(base_dir)
        self.exp_name = exp_name
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.description = description
        # 初始化最优指标记录
        self.best_metric_val = float('inf') if mode == "min" else -float('inf')

        self.run_dir = self._get_next_run_dir()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"🚀 初始化实验目录: {self.run_dir} (监控指标: {monitor_metric})")

    def _get_next_run_dir(self) -> Path:
        """自动寻找当前最大版本号并递增，避免覆盖之前的实验"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        existing_runs = [
            int(p.name.split('_v')[-1])
            for p in self.base_dir.glob(f"{self.exp_name}_v*") if p.is_dir()
        ]
        next_v = max(existing_runs) + 1 if existing_runs else 0
        return self.base_dir / f"{self.exp_name}_v{next_v:03d}"

    def save_hparams(self, hparams: Any, ):
        """
        保存超参数和版本描述。支持直接传入 dataclass 实例。
        """
        # 1. 提取 dataclass 为字典
        if dataclasses.is_dataclass(hparams):
            hparams_dict = dataclasses.asdict(hparams)
        elif isinstance(hparams, dict):
            hparams_dict = hparams
        else:
            raise TypeError("hparams 必须是 dict 或者 dataclass 实例")

        # 2. 组合我们要保存的数据
        save_data = {
            "description": self.description,
            "hparams": hparams_dict
        }

        # 3. 保存为 JSON
        with open(self.run_dir / "hparams.json", "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"📝 超参数与实验描述已保存。")

    def save_ckpt(self, cm_stu, cm_teach, optimizer, step: int, metrics: Dict[str, float],
                  keep_last: Optional[int] = 3):
        """
        保存 checkpoint，记录最优指标，并清理旧版本
        """
        # 1. 准备要保存的 State 字典
        state_dict = {
            "step": step,
            "stu_state": cm_stu.model.state_dict(),
            "teach_state": cm_teach.model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "metrics": metrics
        }

        # 2. 存入当前的 step Checkpoint
        ckpt_path = self.ckpt_dir / f"step_{step}.pt"
        torch.save(state_dict, ckpt_path)

        # 3. 最优指标判断与保存
        current_val = metrics.get(self.monitor_metric)
        is_best = False

        if current_val is not None:
            if (self.mode == "min" and current_val < self.best_metric_val) or \
                    (self.mode == "max" and current_val > self.best_metric_val):
                self.best_metric_val = current_val
                is_best = True

                # 保存 best_model
                best_path = self.ckpt_dir / "best_model.pt"
                torch.save(state_dict, best_path)

                # 单独存一个 best_metrics.json 方便不加载模型直接查看
                with open(self.run_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"step": step, "metrics": metrics}, f, indent=4)

                print(f"🌟 发现新的最优模型! {self.monitor_metric}: {current_val:.4f} (Step: {step})")

        # 4. 自动清理逻辑 (仅清理 step_*.pt，不会影响 best_model.pt)
        if keep_last is not None and keep_last > 0:
            # 找到所有 step 相关的权重文件并按 step 大小排序
            ckpts = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda x: int(x.stem.split('_')[1]))
            # 如果数量超过了 keep_last，删掉最旧的
            for old_ckpt in ckpts[:-keep_last]:
                old_ckpt.unlink()

        return ckpt_path

def convert_to_optical_unet(model: nn.Module, down_num,up_num,optical_kwargs: dict = None):
    """
    将 FlexibleUNet 中的特定卷积层替换为光学卷积层，保持原有参数配置不变。
    """
    if optical_kwargs is None:
        optical_kwargs = {}  # 传入光学层的特有参数，如 noise_std 等

    # ── 1. 替换 input_conv ────────────────────────────────────────────────
    if hasattr(model, 'input_conv') and isinstance(model.input_conv, nn.Conv2d):
        old_conv = model.input_conv
        new_conv = OpticalConv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            **optical_kwargs
        )
        # 拷贝原有权重和偏置
        new_conv.weight.data.copy_(old_conv.weight.data)
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)

        # 重新赋值给模型
        model.input_conv = new_conv

    # ── 2. 替换 down_modules ──────────────────────────────────────────────
    if hasattr(model, 'down_modules'):
        for i, layer in enumerate(model.down_modules):
            if isinstance(layer, nn.Conv2d) and i==down_num:
                new_conv = OpticalConv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=(layer.bias is not None),
                    **optical_kwargs
                )
                new_conv.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    new_conv.bias.data.copy_(layer.bias.data)

                # 更新 ModuleList 中的特定层
                model.down_modules[i] = new_conv

    # ── 3. 替换 up_modules ────────────────────────────────────────────────
    if hasattr(model, 'up_modules'):
        for i, layer in enumerate(model.up_modules):
            if isinstance(layer, nn.ConvTranspose2d) and i==up_num:
                new_tconv = OpticalConvTranspose2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=layer.output_padding,
                    groups=layer.groups,
                    bias=(layer.bias is not None),
                    dilation=layer.dilation,
                    **optical_kwargs
                )
                new_tconv.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    new_tconv.bias.data.copy_(layer.bias.data)

                # 更新 ModuleList 中的特定层
                model.up_modules[i] = new_tconv

    return model


class OpticalCoSimulationEngine:
    """
    光硬件协同仿真管理器，利用 PyTorch Hook 实现无损的特征提取与注入。
    """

    def __init__(self, model: nn.Module, target_layer_types=(OpticalConv2d, OpticalConvTranspose2d)):
        self.model = model
        self.target_layer_types = target_layer_types

        # 存储提取的数据
        self.captured_inputs = {}
        self.captured_sim_outputs = {}

        # 存储需要注入的真实硬件数据
        self.hardware_injections = {}

        # 钩子句柄，用于后期卸载
        self._handles = []

    def _hook_fn(self, layer_name):
        """生成钩子函数：拦截输入、记录原输出、注入新输出"""

        def hook(module, args, output):
            # 1. 窃取：提取并保存输入特征和数字域模拟输出 (必须 detach，防止内存泄漏)
            self.captured_inputs[layer_name] = args[0].detach().cpu()
            self.captured_sim_outputs[layer_name] = output.detach().cpu()

            # 2. 篡改：如果当前层收到了硬件注入指令，则替换输出！
            if layer_name in self.hardware_injections:
                exp_tensor = self.hardware_injections[layer_name]
                # 确保硬件数据的 device 和 dtype 与当前计算图完美对齐
                exp_tensor = exp_tensor.to(device=output.device, dtype=output.dtype)

                # 直接返回注入的张量，PyTorch 会用它替换掉 module 原本的 output，继续向后传播
                return exp_tensor

            # 如果没有注入指令，则正常返回模拟输出
            return output

        return hook

    def register_hooks(self):
        """遍历模型，为所有目标光学层挂载钩子"""
        self.remove_hooks()  # 防止重复挂载
        for name, module in self.model.named_modules():
            # 这里可以根据类名判断，比如 isinstance(module, OpticalConv2d)
            # 假设你已经用之前的方法替换了层，我们直接通过名字或类型识别
            if "input_conv" in name or "down_modules" in name or "up_modules" in name:
                if isinstance(module, self.target_layer_types):
                    handle = module.register_forward_hook(self._hook_fn(name))
                    self._handles.append(handle)
        print(f"✅ Successfully registered hooks on {len(self._handles)} optical layers.")

    def inject_hardware_output(self, layer_name: str, hardware_tensor: torch.Tensor):
        """设置某层的硬件替换数据"""
        self.hardware_injections[layer_name] = hardware_tensor

    def clear_injections(self):
        """清空所有注入数据，恢复纯数字模拟模式"""
        self.hardware_injections.clear()

    def remove_hooks(self):
        """卸载所有钩子，完全恢复模型自由身"""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.captured_inputs.clear()
        self.captured_sim_outputs.clear()
