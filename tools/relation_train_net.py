# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.solver import make_ship_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


# See if we can use apex.DistributedDataParallel instead of the torch default,

from torch.cuda.amp import autocast as autocast, GradScaler
from thop import  profile

def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg) 
    debug_print(logger, 'end model construction')
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
 
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []
    
    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    
    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    ship_optimizer = make_ship_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, 
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        ship_optimizer_state = extra_checkpoint_data.pop("ship_optimizer", None)
        if ship_optimizer is not None and ship_optimizer_state is not None:
            ship_optimizer.load_state_dict(ship_optimizer_state)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    test_data_loaders = None
    if cfg.SOLVER.TO_VAL:
        test_data_loaders = {
            "novel": build_test_data_loaders(cfg, distributed, "novel"),
            "base": build_test_data_loaders(cfg, distributed, "base"),
        }
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Evaluation before training is disabled")

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    last_eval_iteration = None
    
    scaler = GradScaler()
    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        tmp=0
        for data in targets:

            if data.get_field("relation").shape[0]==1:
                print(data.get_field("relation"))
                tmp=1
        if tmp==1:
            continue
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        loss_dict = model(images, targets)


        losses = sum(loss for loss in loss_dict.values())
        #print(losses)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        if ship_optimizer is not None:
            ship_optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe

        losses.backward()
        
        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()
        if ship_optimizer is not None:
            ship_optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0 and iteration>=12000:
            checkpointer.save(
                "model_{:07d}".format(iteration),
                ship_optimizer=ship_optimizer.state_dict() if ship_optimizer is not None else None,
                **arguments
            )
        if iteration == max_iter :
            checkpointer.save(
                "model_final",
                ship_optimizer=ship_optimizer.state_dict() if ship_optimizer is not None else None,
                **arguments
            )

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0 and iteration>=8000:
            logger.info("Start evaluating test-novel")
            novel_result = run_test(
                cfg, model, test_data_loaders["novel"], distributed, logger, "novel"
            )
            logger.info("Test Novel Result: %.4f" % novel_result)
            logger.info("Start evaluating test-base")
            base_result = run_test(
                cfg, model, test_data_loaders["base"], distributed, logger, "base"
            )
            logger.info("Test Base Result: %.4f" % base_result)
            val_result = base_result
            last_eval_iteration = iteration

             
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model, last_eval_iteration == arguments["iteration"]

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def build_test_data_loaders(cfg, distributed, eval_part):
    """Build test data once with GT filtered for one predicate subset."""
    if eval_part not in {"base", "novel", "total", "semantic"}:
        raise ValueError("Unsupported test predicate part: {}".format(eval_part))

    # VGDataset reads the process-global cfg during construction.
    original_test_part = cfg.OV_SETTING.TEST_PART
    cfg.defrost()
    cfg.OV_SETTING.TEST_PART = eval_part
    try:
        return make_data_loader(
            cfg,
            mode="test",
            is_distributed=distributed,
        )
    finally:
        cfg.OV_SETTING.TEST_PART = original_test_part
        cfg.freeze()


def run_test(cfg, model, test_data_loaders, distributed, logger, eval_part):
    if distributed:
        model = model.module

    torch.cuda.empty_cache()
    model.updata(eval_part)

    eval_cfg = cfg.clone()
    eval_cfg.defrost()
    eval_cfg.OV_SETTING.TEST_PART = eval_part
    eval_cfg.freeze()

    iou_types = ("bbox",)
    if eval_cfg.MODEL.MASK_ON:
        iou_types += ("segm",)
    if eval_cfg.MODEL.KEYPOINT_ON:
        iou_types += ("keypoints",)
    if eval_cfg.MODEL.RELATION_ON:
        iou_types += ("relations",)
    if eval_cfg.MODEL.ATTRIBUTE_ON:
        iou_types += ("attributes",)

    dataset_names = eval_cfg.DATASETS.TEST
    output_folders = [None] * len(dataset_names)
    if eval_cfg.OUTPUT_DIR:
        for index, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(
                eval_cfg.OUTPUT_DIR,
                "inference",
                eval_part,
                dataset_name,
            )
            mkdir(output_folder)
            output_folders[index] = output_folder

    results = []
    for output_folder, dataset_name, data_loader in zip(
        output_folders, dataset_names, test_data_loaders
    ):
        results.append(
            inference(
                eval_cfg,
                model,
                data_loader,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if eval_cfg.MODEL.RETINANET_ON else eval_cfg.MODEL.RPN_ONLY,
                device=eval_cfg.MODEL.DEVICE,
                expected_results=eval_cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=eval_cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                logger=logger,
            )
        )
        synchronize()

    gathered = all_gather(torch.tensor(results).cpu())
    gathered = torch.cat([value.view(-1) for value in gathered], dim=-1).view(-1)
    valid = gathered[gathered >= 0]
    result = float(valid.mean())
    del gathered, valid
    torch.cuda.empty_cache()
    model.updata(cfg.OV_SETTING.TRAIN_PART)
    return result


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model, final_eval_done = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test and not final_eval_done:
        novel_loaders = build_test_data_loaders(cfg, args.distributed, "novel")
        novel_result = run_test(
            cfg, model, novel_loaders, args.distributed, logger, "novel"
        )
        logger.info("Test Novel Result: %.4f" % novel_result)

        base_loaders = build_test_data_loaders(cfg, args.distributed, "base")
        base_result = run_test(
            cfg, model, base_loaders, args.distributed, logger, "base"
        )
        logger.info("Test Base Result: %.4f" % base_result)


if __name__ == "__main__":
    main()
