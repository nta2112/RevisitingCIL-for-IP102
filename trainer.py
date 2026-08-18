import sys
import logging
import copy
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.metrics import (
    evaluate_retrieval,
    evaluate_ood,
    extract_embeddings,
    compute_lifelong,
)
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    history = {}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        ret_metrics = _eval_task_metrics(data_manager, model, args, history)
        _write_result_csv(
            os.path.join(logs_name, "results.csv"),
            task,
            model._total_classes,
            cnn_accy,
            nme_accy,
            ret_metrics,
        )
        _write_history_json(os.path.join(logs_name, "history.json"), history)

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _write_history_json(json_path, history):
    import json

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _eval_task_metrics(data_manager, model, args, history):
    task_id = model._cur_task
    seen = list(range(model._total_classes))
    all_classes = list(range(data_manager.get_total_classnum()))
    device = args["device"][0]
    val_source = "val" if data_manager._val_data is not None else "test"

    val_ds = data_manager.get_dataset(seen, source=val_source, mode="test")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = model.test_loader

    q_emb, q_cls = extract_embeddings(model._network, val_loader, device)
    g_emb, g_cls = extract_embeddings(model._network, test_loader, device)

    _, macro_r1, macro_r5, macro_r10, macro_ap, q_aps, q_r1s = evaluate_retrieval(
        q_emb, q_cls, g_emb, g_cls
    )

    task_of_class = {}
    offset = 0
    for t, sz in enumerate(data_manager._increments):
        for c in range(offset, offset + sz):
            task_of_class[c] = t
        offset += sz

    task_ids = sorted(set(task_of_class[c] for c in seen))
    group_map = {}
    for t in task_ids:
        mask = np.array([task_of_class[c] == t for c in q_cls])
        if mask.sum() > 0:
            group_map[t] = float(np.mean(q_aps[mask]))

    history[str(task_id)] = {str(t): round(group_map[t], 6) for t in group_map}
    plasticity, forgetting, overall = compute_lifelong(group_map, task_id, history)

    val_all_ds = data_manager.get_dataset(all_classes, source=val_source, mode="test")
    val_all_loader = DataLoader(
        val_all_ds, batch_size=64, shuffle=False, num_workers=4
    )
    q_all_emb, q_all_cls = extract_embeddings(model._network, val_all_loader, device)

    seen_class_means = np.stack([g_emb[g_cls == c].mean(0) for c in seen])
    auroc, fpr95 = evaluate_ood(q_all_emb, q_all_cls, seen_class_means, seen)

    logging.info(
        "Retrieval (seen classes): R@1 {:.3f} | R@5 {:.3f} | R@10 {:.3f} | mAP {:.3f}".format(
            macro_r1, macro_r5, macro_r10, macro_ap
        )
    )
    logging.info(
        "OOD AUROC {:.3f} | FPR@TPR95 {:.3f}".format(
            (auroc or 0.0), (fpr95 or 0.0)
        )
    )
    logging.info(
        "Plasticity {:.3f} | Forgetting {:.3f} | Overall {:.3f}".format(
            plasticity, forgetting, overall
        )
    )

    return {
        "r1": macro_r1,
        "r5": macro_r5,
        "r10": macro_r10,
        "mAP": macro_ap,
        "auroc": auroc,
        "fpr95": fpr95,
        "plasticity": plasticity,
        "forgetting": forgetting,
        "overall": overall,
    }


def _write_result_csv(csv_path, task, numclass, cnn_accy, nme_accy, m):
    header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(
                [
                    "task",
                    "numclass",
                    "cnn_top1",
                    "nme_top1",
                    "R@1",
                    "R@5",
                    "R@10",
                    "mAP",
                    "AUROC",
                    "FPR95",
                    "Plasticity",
                    "Forgetting",
                    "Overall",
                ]
            )
        w.writerow(
            [
                task,
                numclass,
                cnn_accy["top1"] if cnn_accy is not None else "-",
                nme_accy["top1"] if nme_accy is not None else "-",
                round(float(m["r1"]), 3),
                round(float(m["r5"]), 3),
                round(float(m["r10"]), 3),
                round(float(m["mAP"]), 3),
                "-" if m["auroc"] is None else round(float(m["auroc"]), 3),
                "-" if m["fpr95"] is None else round(float(m["fpr95"]), 3),
                round(float(m["plasticity"]), 3),
                round(float(m["forgetting"]), 3),
                round(float(m["overall"]), 3),
            ]
        )
