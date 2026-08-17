# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau


def _collect_ship_parameters(model):
    parameters = []
    parameter_ids = set()
    for module in model.modules():
        provider = getattr(module, "ship_parameters", None)
        if not callable(provider):
            continue
        for parameter in provider():
            if parameter.requires_grad and id(parameter) not in parameter_ids:
                parameters.append(parameter)
                parameter_ids.add(id(parameter))
    return parameters, parameter_ids


def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0):
    params = []
    total=0
    _, ship_parameter_ids = _collect_ship_parameters(model)
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if id(value) in ship_parameter_ids:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "roi_heads.relation.predictor.adaper_clip"in key:

            total+=value.nelement()
            print(total/1e6)
        if slow_heads is not None:
            for item in slow_heads:
                if item in key:
                    logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                    lr = lr / slow_ratio
                    break
        params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)

    #optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR,eps=1e-4)
    return optimizer


def make_mtm_ship_optimizer(cfg, model):
    mtm_cfg = cfg.MODEL.ROI_RELATION_HEAD.MTM
    if not (
        mtm_cfg.ENABLED
        and mtm_cfg.TRAINING_ENABLED
        and mtm_cfg.SHIP.ENABLED
    ):
        return None
    parameters, _ = _collect_ship_parameters(model)
    if len(parameters) == 0:
        return None
    optimizer_config = mtm_cfg.SHIP.OPTIMIZER
    return torch.optim.AdamW(
        parameters,
        lr=optimizer_config.LR,
        weight_decay=optimizer_config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )


def validate_optimizer_parameters(model, *optimizers):
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assigned = []
    for optimizer in optimizers:
        if optimizer is None:
            continue
        assigned.extend(
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
    if len(assigned) != len(set(assigned)):
        raise RuntimeError("A trainable parameter belongs to multiple optimizers")
    if set(assigned) != expected:
        missing = len(expected - set(assigned))
        unexpected = len(set(assigned) - expected)
        raise RuntimeError(
            "Optimizer parameter coverage mismatch: missing={}, unexpected={}".format(
                missing, unexpected
            )
        )


def make_lr_scheduler(cfg, optimizer, logger=None):
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    
    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )
    
    else:
        raise ValueError("Invalid Schedule Type")
