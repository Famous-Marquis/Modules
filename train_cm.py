from dataclasses import dataclass,field
from typing import Tuple, List,Union,Dict,Any

import torch
import torchvision
from torchvision import transforms
from models import Consistency, FlexibleUNet, train_cm, make_loss_f
from utils import RunManager, convert_to_optical_unet


@dataclass
class ConsistencyConfig:
    # training set
    max_epochs: int = 200
    batch_size: int = 128
    eval_num: int = 10000
    mse_w: float = 0.
    huber_w: float = 0.
    c: float = 0.01  # 0.00054~0.03
    lpips_w: float = 1.
    lambda_w: float = 1.

    # model arch
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    # 列表是可变对象，在 dataclass 中必须用 default_factory
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 4
    use_attention: Union[bool, List[bool]] = False
    time_emb_dim: int = 128
    time_mlp_ratio: int = 4
    dropout: float = 0.1
    norm_groups: int = 8
    middle_attention: bool = False
    continuous_time:bool=True,

    # cm set
    min_discrete: int = 10
    max_discrete: int = 150
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80
    rho: float = 7.0
    mu0: float = 0.95
    lr: float = 1e-4
    # 元组是不可变对象，可以直接赋值
    input_img_shape: Tuple[int, int, int] = (1, 32, 32)
    # 2. 定义你想注入的光学参数
    down_num: int = 0
    up_num:int=1
    optical_params: Dict[str, Any] = field(default_factory=lambda: {
        'weight_noise_std': 0.03,
        'prev_noise_std': 0.05,
        'post_noise_std': 0.05,
        'weight_clip': 1.0,
        'apply_sn': True
    })
if __name__ == '__main__':
    cfg = ConsistencyConfig()

    run_manager = RunManager(
        description="lpips loss",# 下一个huber loss
        base_dir="./runs",
        exp_name="consistency-fashion-robust",
        monitor_metric="fid",
        mode="min"
    )

    run_manager.save_hparams(hparams=cfg)

    model_stu, model_teach = (FlexibleUNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels,
                                           base_channels=cfg.base_channels,
                                           channel_multipliers=cfg.channel_multipliers,
                                           num_res_blocks=cfg.num_res_blocks,
                                           use_attention=cfg.use_attention,
                                           time_emb_dim=cfg.time_emb_dim, time_mlp_ratio=cfg.time_mlp_ratio,
                                           dropout=cfg.dropout, norm_groups=cfg.norm_groups,
                                           middle_attention=cfg.middle_attention,continuous_time=cfg.continuous_time),
                              FlexibleUNet(in_channels=cfg.in_channels, out_channels=cfg.out_channels,
                                           base_channels=cfg.base_channels,
                                           channel_multipliers=cfg.channel_multipliers,
                                           num_res_blocks=cfg.num_res_blocks,
                                           use_attention=cfg.use_attention,
                                           time_emb_dim=cfg.time_emb_dim, time_mlp_ratio=cfg.time_mlp_ratio,
                                           dropout=cfg.dropout, norm_groups=cfg.norm_groups,
                                           middle_attention=cfg.middle_attention,continuous_time=cfg.continuous_time))

    convert_to_optical_unet(model_stu,down_num=cfg.down_num,up_num=cfg.up_num)
    convert_to_optical_unet(model_teach,down_num=cfg.down_num,up_num=cfg.up_num)

    cm_stu, cm_teach = (Consistency(model_stu, min_discrete=cfg.min_discrete, max_discrete=cfg.max_discrete,
                                   sigma_data=cfg.sigma_data, sigma_min=cfg.sigma_min,
                                   sigma_max=cfg.sigma_max, rho=cfg.rho, mu0=cfg.mu0, lr=cfg.lr,
                                   input_img_shape=cfg.input_img_shape),
                        Consistency(model_teach, min_discrete=cfg.min_discrete, max_discrete=cfg.max_discrete,
                                   sigma_data=cfg.sigma_data, sigma_min=cfg.sigma_min,
                                   sigma_max=cfg.sigma_max, rho=cfg.rho, mu0=cfg.mu0, lr=cfg.lr,
                                   input_img_shape=cfg.input_img_shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_stu.model,cm_teach.model=cm_stu.model.to(device),cm_teach.model.to(device)
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                                     ]))
    # dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
    #                                             transform=transforms.Compose(
    #                                                 [transforms.ToTensor(),
    #                                                 transforms.Normalize(mean=[0.5], std=[0.5])
    #                                                  ]))
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True,
                                             num_workers=4)

    # print(cm_stu.generate(64).shape)
    loss_fn=make_loss_f(mse_w=cfg.mse_w,huber_w=cfg.huber_w,lpips_w=cfg.lpips_w,c=cfg.c,lambda_w=cfg.lambda_w)
    train_cm(log_manager=run_manager, max_epoch=cfg.max_epochs, dataloader=dataloader, cm_teach=cm_teach,
             cm_stu=cm_stu, eval_num=cfg.eval_num,device=device,loss_fn=loss_fn)
    # print(model_stu)
