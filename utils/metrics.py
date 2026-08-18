import numpy as np
import torch


def _l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-8)


def calculate_auroc_numpy(y_true, y_scores):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_score_indices]
    y_scores = y_scores[desc_score_indices]

    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))

    area = 0.0
    for i in range(len(fpr) - 1):
        area += 0.5 * (tpr[i] + tpr[i + 1]) * (fpr[i + 1] - fpr[i])
    return area, fpr, tpr


def calculate_fpr_at_tpr95(fpr, tpr):
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    return 1.0


def compute_ap(q_class, ranked_classes):
    ap = 0.0
    hits = 0
    total_pos = sum(1 for g_class in ranked_classes if g_class == q_class)
    if total_pos == 0:
        return 0.0
    for rank, g_class in enumerate(ranked_classes):
        if g_class == q_class:
            hits += 1
            precision = hits / (rank + 1)
            ap += precision
            if hits == total_pos:
                break
    return ap / total_pos


def evaluate_retrieval(q_embeds, q_classes, g_embeds, g_classes):
    """Cosine-similarity retrieval.

    Returns (class_metrics, macro_r1, macro_r5, macro_r10, macro_ap, q_aps, q_r1s)
    where q_aps / q_r1s are per-query arrays aligned with the query order.
    """
    q_embeds = _l2_normalize(q_embeds)
    g_embeds = _l2_normalize(g_embeds)
    q_classes = np.asarray(q_classes)
    g_classes = np.asarray(g_classes)

    sims = q_embeds @ g_embeds.T
    n = len(q_embeds)
    r1s = np.zeros(n)
    r5s = np.zeros(n)
    r10s = np.zeros(n)
    aps = np.zeros(n)

    for i in range(n):
        ranked = g_classes[np.argsort(sims[i])[::-1]]
        qc = q_classes[i]
        r1s[i] = 1.0 if qc in ranked[:1] else 0.0
        r5s[i] = 1.0 if qc in ranked[:5] else 0.0
        r10s[i] = 1.0 if qc in ranked[:10] else 0.0
        aps[i] = compute_ap(qc, ranked)

    class_metrics = {}
    for c in np.unique(q_classes):
        mask = q_classes == c
        class_metrics[int(c)] = {
            'count': int(mask.sum()),
            'R@1': float(np.mean(r1s[mask])),
            'R@5': float(np.mean(r5s[mask])),
            'R@10': float(np.mean(r10s[mask])),
            'AP': float(np.mean(aps[mask])),
        }

    nc = len(class_metrics)
    if nc == 0:
        return {}, 0.0, 0.0, 0.0, 0.0, aps, r1s
    macro_r1 = np.mean([class_metrics[c]['R@1'] for c in class_metrics])
    macro_r5 = np.mean([class_metrics[c]['R@5'] for c in class_metrics])
    macro_r10 = np.mean([class_metrics[c]['R@10'] for c in class_metrics])
    macro_ap = np.mean([class_metrics[c]['AP'] for c in class_metrics])
    return class_metrics, float(macro_r1), float(macro_r5), float(macro_r10), float(macro_ap), aps, r1s


def evaluate_ood(q_embeds, q_classes, seen_class_means, seen_classes):
    """Open-set anomaly detection: unseen classes are 'unknown'.

    Unknown score = -max cosine similarity to any seen-class mean.
    Returns (auroc, fpr95) or (None, None) if a class is missing.
    """
    q_embeds = _l2_normalize(q_embeds)
    seen_class_means = _l2_normalize(np.asarray(seen_class_means))
    sims = q_embeds @ seen_class_means.T
    max_sim = sims.max(axis=1)
    unknown_score = -max_sim

    seen_set = set(int(c) for c in seen_classes)
    y_true = np.array([0 if int(c) in seen_set else 1 for c in q_classes])

    num_pos = int(y_true.sum())
    num_neg = len(y_true) - num_pos
    if num_pos > 0 and num_neg > 0:
        auroc, fpr, tpr = calculate_auroc_numpy(y_true, unknown_score)
        fpr95 = calculate_fpr_at_tpr95(fpr, tpr)
        return float(auroc), float(fpr95)
    return None, None


def extract_embeddings(network, loader, device):
    """Extract normalized feature embeddings from a network over a loader.

    Handles SimpleVitNet (extract_vector returns a raw tensor) and
    MultiBranch nets (extract_vector unavailable -> forward 'features').
    """
    network.eval()
    embeds, labels = [], []
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device)
            feats = None
            if hasattr(network, "extract_vector"):
                try:
                    feats = network.extract_vector(inputs)
                except Exception:
                    feats = None
            if feats is None:
                out = network(inputs)
                feats = out.get("features", out.get("logits"))
            if isinstance(feats, dict):
                feats = feats.get("features", feats.get("logits"))
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
            embeds.append(feats)
            labels.append(targets.numpy())
    return np.concatenate(embeds), np.concatenate(labels)


def compute_lifelong(group_map, task_id, history):
    """Plasticity / Forgetting / Overall from per-task group mAP history.

    history: dict task_id -> {old_task_id: group_mAP}.
    """
    plasticity = group_map.get(task_id, 0.0)
    forgets = []
    for t in range(task_id):
        if str(t) in history and str(task_id) in history:
            peak = history[str(t)].get(str(t), 0.0)
            curr = history[str(task_id)].get(str(t), 0.0)
            forgets.append(max(0.0, peak - curr))
    forgetting = float(np.mean(forgets)) if forgets else 0.0
    overall = plasticity - forgetting
    return plasticity, forgetting, overall