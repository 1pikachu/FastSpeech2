import argparse
import os
import time

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    args.datatype = torch.float16 if args.precision == "float16" else \
        torch.bfloat16 if args.precision == "bfloat16" else torch.float32

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, args, sort=True, drop_last=True
    )
    #batch_size = train_config["optimizer"]["batch_size"]
    batch_size = args.batch_size
    group_size = 1  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    batch_size = batch_size * group_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, args.device, train=True)
    if args.device == "xpu":
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=args.datatype)
    #model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(args.device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, args.device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    # step = args.restore_step + 1
    step = 0
    epoch = 1
    #grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_acc_step = 1
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    total_step = args.num_iters
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    total_time = 0.0
    total_count = 0
    profile_len = min(len(loader), args.num_iters) // 2

    while True:
        if args.profile and args.device == "xpu":
            for batchs in loader:
                for batch in batchs:
                    with torch.autograd.profiler_legacy.profile(args.profile, use_xpu=True) as prof:
                        start_time = time.time()
                        batch = to_device(batch, args.device)

                        # Forward
                        output = model(*(batch[2:]))

                        # Cal Loss
                        losses = Loss(batch, output)
                        total_loss = losses[0]

                        # Backward
                        total_loss = total_loss / grad_acc_step
                        total_loss.backward()

                        if step % grad_acc_step == 0:
                            # Clipping gradients to avoid gradient explosion
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                            # Update weights
                            optimizer.step_and_update_lr()
                            optimizer.zero_grad()
                        torch.xpu.synchronize()
                        end_time = time.time()
                    if step >= args.num_warmup:
                        total_time += end_time - start_time
                        total_count += 1
                    print("iteration:{}, training time: {} sec.".format(step, end_time - start_time))
                    if cfg.model.oob_profile and i == profile_len:
                        import pathlib
                        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                        if not os.path.exists(timeline_dir):
                            try:
                                os.makedirs(timeline_dir)
                            except:
                                pass
                        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                            timeline_dir+'profile.pt')
                        torch.save(prof.key_averages(group_by_input_shape=True).table(),
                            timeline_dir+'profile_detail.pt')
                        torch.save(prof.table(sort_by="id", row_limit=100000),
                            timeline_dir+'profile_detail_withId.pt')
                        prof.export_chrome_trace(timeline_dir+"trace.json")

                    if step == total_step:
                        avg_time = total_time / total_count
                        latency = avg_time / batch_size * 1000
                        perf = batch_size / avg_time
                        print("total time:{}, total count:{}".format(total_time, total_count))
                        print('%d epoch training latency: %6.2f ms'%(epoch, latency))
                        print('%d epoch training Throughput: %6.2f fps'%(epoch, perf))
                        quit()
                    step += 1
        elif args.profile and args.device == "cuda":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_len,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for batchs in loader:
                    for batch in batchs:
                        start_time = time.time()
                        batch = to_device(batch, args.device)

                        # Forward
                        output = model(*(batch[2:]))

                        # Cal Loss
                        losses = Loss(batch, output)
                        total_loss = losses[0]

                        # Backward
                        total_loss = total_loss / grad_acc_step
                        total_loss.backward()

                        if step % grad_acc_step == 0:
                            # Clipping gradients to avoid gradient explosion
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                            # Update weights
                            optimizer.step_and_update_lr()
                            optimizer.zero_grad()
                        torch.cuda.synchronize()
                        end_time = time.time()
                        p.step()
                        if step >= args.num_warmup:
                            total_time += end_time - start_time
                            total_count += 1
                        print("iteration:{}, training time: {} sec.".format(step, end_time - start_time))

                        if step == total_step:
                            avg_time = total_time / total_count
                            latency = avg_time / batch_size * 1000
                            perf = batch_size / avg_time
                            print("total time:{}, total count:{}".format(total_time, total_count))
                            print('%d epoch training latency: %6.2f ms'%(epoch, latency))
                            print('%d epoch training Throughput: %6.2f fps'%(epoch, perf))
                            quit()
                        step += 1
        else:
            for batchs in loader:
                for batch in batchs:
                    start_time = time.time()
                    batch = to_device(batch, args.device)

                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    # Backward
                    total_loss = total_loss / grad_acc_step
                    total_loss.backward()

                    if step % grad_acc_step == 0:
                        # Clipping gradients to avoid gradient explosion
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                        # Update weights
                        optimizer.step_and_update_lr()
                        optimizer.zero_grad()
                    if args.device == "xpu":
                        torch.xpu.synchronize()
                    elif args.device == "cuda":
                        torch.cuda.synchronize()
                    end_time = time.time()
                    if step >= args.num_warmup:
                        total_time += end_time - start_time
                        total_count += 1
                    print("iteration:{}, training time: {} sec.".format(step, end_time - start_time))

                    #if step % log_step == 0:
                    #    losses = [l.item() for l in losses]
                    #    message1 = "Step {}/{}, ".format(step, total_step)
                    #    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    #        *losses
                    #    )

                    #    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    #        f.write(message1 + message2 + "\n")

                    #    log(train_logger, step, losses=losses)

                    #if step % synth_step == 0:
                    #    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                    #        batch,
                    #        output,
                    #        vocoder,
                    #        model_config,
                    #        preprocess_config,
                    #    )
                    #    log(
                    #        train_logger,
                    #        fig=fig,
                    #        tag="Training/step_{}_{}".format(step, tag),
                    #    )
                    #    sampling_rate = preprocess_config["preprocessing"]["audio"][
                    #        "sampling_rate"
                    #    ]
                    #    log(
                    #        train_logger,
                    #        audio=wav_reconstruction,
                    #        sampling_rate=sampling_rate,
                    #        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    #    )
                    #    log(
                    #        train_logger,
                    #        audio=wav_prediction,
                    #        sampling_rate=sampling_rate,
                    #        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    #    )

                    #if step % val_step == 0:
                    #    model.eval()
                    #    message = evaluate(model, step, configs, val_logger, vocoder)
                    #    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    #        f.write(message + "\n")
                    #    outer_bar.write(message)

                    #    model.train()

                    #if step % save_step == 0:
                    #    torch.save(
                    #        {
                    #            "model": model.module.state_dict(),
                    #            "optimizer": optimizer._optimizer.state_dict(),
                    #        },
                    #        os.path.join(
                    #            train_config["path"]["ckpt_path"],
                    #            "{}.pth.tar".format(step),
                    #        ),
                    #    )

                    if step == total_step:
                        avg_time = total_time / total_count
                        latency = avg_time / batch_size * 1000
                        perf = batch_size / avg_time
                        print("total time:{}, total count:{}".format(total_time, total_count))
                        print('%d epoch training latency: %6.2f ms'%(epoch, latency))
                        print('%d epoch training Throughput: %6.2f fps'%(epoch, perf))
                        quit()
                    step += 1
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    # OOB
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iters', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    args = parser.parse_args()
    print(args)

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
