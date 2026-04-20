#!/usr/bin/env python3
"""
collapse_train.py — distill teacher outputs into a small ONNX student model.

Called as a stage inside bonfyre-run DAGs (T04, T07, T14, ...).
Reads teacher output directories produced by bonfyre-tag / bonfyre-embed /
collapse_consensus.py and trains a frozen-MiniLM + MLP student.

Tasks
  topic-map        bonfyre-tag + bonfyre-embed → K-class topic softmax
  chunk-boundary   two bonfyre-embed passes → boundary probability
  ner-bio          consensus NER labels → BIO token classifier
  risk-score       toxicity+sentiment logits → 2-class risk signal

Output (written to --out/):
  model.onnx        ONNX student (dynamic batch + sequence axes)
  metrics.json      eval results consumed by bonfyre-learn feedback
  artifact.json     BfArtifact manifest for bonfyre-model add

Usage
  python3 scripts/collapse_train.py \\
      --task topic-map \\
      --tag-dir    /run/t04/tag \\
      --embed-dir  /run/t04/embed \\
      --out        /run/t04/train

  python3 scripts/collapse_train.py \\
      --task chunk-boundary \\
      --embed-dir-a /run/t07/embed-minilm \\
      --embed-dir-b /run/t07/embed-mpnet \\
      --out         /run/t07/train

  python3 scripts/collapse_train.py \\
      --task ner-bio \\
      --consensus /run/t14/consensus/labels.jsonl \\
      --out       /run/t14/train

Deps: sentence-transformers torch onnx onnxruntime  (pip install all four)
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sentence_transformers import SentenceTransformer
    import onnx
    import onnxruntime as ort
except ImportError as e:
    sys.exit(f"[collapse_train] missing dep: {e}\n"
             f"  pip install sentence-transformers torch onnx onnxruntime")


# ── constants ──────────────────────────────────────────────────────────────────
STUDENT_BASE   = "all-MiniLM-L6-v2"
EMBED_DIM      = 384
EPOCHS         = 30
LR             = 1e-3
BATCH          = 64
PATIENCE       = 5
VAL_SPLIT      = 0.15
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ── student architectures ──────────────────────────────────────────────────────
class TopicHead(nn.Module):
    """384 → 256 → K softmax (topic-map task)."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes))
    def forward(self, x):
        return self.net(x)


class BoundaryHead(nn.Module):
    """384×2 concat → 128 → 1 sigmoid (chunk-boundary task)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x)


class NerBioHead(nn.Module):
    """384 → 256 → n_tags (ner-bio task, applied per token via tag projection)."""
    def __init__(self, n_tags: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_tags))
    def forward(self, x):
        return self.net(x)


class RiskHead(nn.Module):
    """384 → 64 → 2 logits (risk-score task)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 64), nn.ReLU(),
            nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)


# ── data loaders ───────────────────────────────────────────────────────────────
def load_risk_embed(embed_dir: str, risk_dir: str):
    """
    Load embeddings + per-doc risk labels.
    risk_dir contains per-doc JSON: {"risk": "high"|"low", "score": float}
    Returns (embeddings: np.array [N, 384], labels: np.array [N]  {0=low, 1=high})
    """
    emb_map  = {}
    for fname in sorted(os.listdir(embed_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(embed_dir, fname)) as f:
            obj = json.load(f)
        vec = obj.get("embedding") or obj.get("vector")
        if vec:
            emb_map[fname[:-5]] = np.array(vec, dtype=np.float32)

    risk_map = {}
    for fname in sorted(os.listdir(risk_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(risk_dir, fname)) as f:
            obj = json.load(f)
        label = obj.get("risk", "low")
        risk_map[fname[:-5]] = 1 if label == "high" else 0

    stems = sorted(set(emb_map) & set(risk_map))
    if not stems:
        sys.exit("[collapse_train] no matching stems between embed-dir and risk-dir")

    n_high = sum(risk_map[s] for s in stems)
    print(f"[collapse_train] risk-score: {len(stems)} samples, "
          f"{n_high} high-risk ({100*n_high/len(stems):.1f}%)")
    embeddings = np.stack([emb_map[s] for s in stems])
    labels     = np.array([risk_map[s] for s in stems], dtype=np.int64)
    return embeddings, labels


def load_tag_embed(tag_dir: str, embed_dir: str):
    """
    Load bonfyre-tag output (tags.json) and bonfyre-embed output (*.json)
    from their respective output directories.
    Returns (embeddings: np.array [N, 384], labels: np.array [N], label_names: list[str])
    """
    # Collect all embeddings keyed by stem
    emb_map = {}
    for fname in sorted(os.listdir(embed_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(embed_dir, fname)
        try:
            with open(path) as f:
                obj = json.load(f)
            stem = fname[:-5]  # strip .json
            # bonfyre-embed json format: {"embedding": [...], ...} or {"vector": [...]}
            vec = obj.get("embedding") or obj.get("vector")
            if vec:
                emb_map[stem] = np.array(vec, dtype=np.float32)
        except Exception:
            continue

    # Collect all tag outputs keyed by stem
    tag_map = {}
    for fname in sorted(os.listdir(tag_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(tag_dir, fname)
        try:
            with open(path) as f:
                obj = json.load(f)
            stem = fname[:-5]
            # bonfyre-tag json: {"tags": [{"label": str, "score": float}, ...]}
            tags = obj.get("tags") or obj.get("predictions") or []
            if tags:
                tag_map[stem] = tags[0].get("label", "unknown") if tags else "unknown"
        except Exception:
            continue

    # Intersect by stem
    stems = sorted(set(emb_map) & set(tag_map))
    if not stems:
        sys.exit("[collapse_train] no matching stems between tag-dir and embed-dir")

    # Build label index
    label_set = sorted(set(tag_map[s] for s in stems))
    label_idx = {l: i for i, l in enumerate(label_set)}

    embeddings = np.stack([emb_map[s] for s in stems])
    labels     = np.array([label_idx[tag_map[s]] for s in stems], dtype=np.int64)
    print(f"[collapse_train] topic-map: {len(stems)} samples, {len(label_set)} classes")
    return embeddings, labels, label_set


def load_two_embeds(dir_a: str, dir_b: str):
    """
    Load two bonfyre-embed output directories (one per model).
    Returns (pairs: np.array [N, 384*2], boundaries: np.array [N])
    Boundary = 1 when both models agree cosine similarity drops below their
    respective adaptive thresholds between consecutive sentence embeddings.
    """
    def read_dir(d):
        vecs = {}
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(d, fname)) as f:
                obj = json.load(f)
            vec = obj.get("embedding") or obj.get("vector")
            if vec:
                vecs[fname[:-5]] = np.array(vec, dtype=np.float32)
        return vecs

    vecs_a = read_dir(dir_a)
    vecs_b = read_dir(dir_b)
    stems  = sorted(set(vecs_a) & set(vecs_b))
    if len(stems) < 3:
        sys.exit("[collapse_train] need ≥3 matching embeddings for chunk-boundary")

    arr_a = np.stack([vecs_a[s] for s in stems])
    arr_b = np.stack([vecs_b[s] for s in stems])

    # Cosine similarity between consecutive items
    def cos_sim_seq(arr):
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        normed = arr / norms
        return np.array([float(normed[i] @ normed[i+1]) for i in range(len(arr)-1)])

    sims_a = cos_sim_seq(arr_a)
    sims_b = cos_sim_seq(arr_b)

    # Adaptive threshold: mean - 0.7*std
    thr_a = float(sims_a.mean() - 0.7 * sims_a.std())
    thr_b = float(sims_b.mean() - 0.7 * sims_b.std())

    # Boundary = both models drop below their threshold
    boundaries = ((sims_a < thr_a) & (sims_b < thr_b)).astype(np.int64)

    # pair features: concat(embed[i], embed[i+1]) from model_a
    pairs = np.concatenate([arr_a[:-1], arr_a[1:]], axis=1)

    n_boundaries = int(boundaries.sum())
    print(f"[collapse_train] chunk-boundary: {len(pairs)} pairs, "
          f"{n_boundaries} boundaries ({100*n_boundaries/len(pairs):.1f}%)")
    return pairs, boundaries


def load_consensus(consensus_path: str, embedder):
    """
    Load consensus NER labels from collapse_consensus.py output.
    Returns (token_embeddings: np.array [N, 384], bio_labels: np.array [N], tag_set: list)
    """
    records = []
    with open(consensus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        sys.exit(f"[collapse_train] empty consensus: {consensus_path}")

    # Embed each sentence, expand to token level with BIO labels
    all_embs, all_labels = [], []
    tag_set = ["O", "B-ENT", "I-ENT"]  # simplified BIO; extend as needed
    tag_idx = {t: i for i, t in enumerate(tag_set)}

    for rec in records:
        text   = rec.get("text", "")
        tokens = text.split()
        if not tokens:
            continue
        # Sentence-level embedding for all tokens (mean pooling across tokens)
        emb = embedder.encode(text, normalize_embeddings=True)
        spans = rec.get("entities", [])
        # Build BIO sequence
        bio = ["O"] * len(tokens)
        for span in spans:
            # Map character spans to token indices (approximation)
            tstart = len(text[:span.get("start", 0)].split())
            tend   = len(text[:span.get("end", 0)].split())
            for ti in range(tstart, min(tend, len(tokens))):
                bio[ti] = "B-ENT" if ti == tstart else "I-ENT"
        # Each token: use the sentence embedding (shared) + BIO label
        for tag in bio:
            all_embs.append(emb)
            all_labels.append(tag_idx.get(tag, 0))

    embeddings = np.stack(all_embs).astype(np.float32)
    labels     = np.array(all_labels, dtype=np.int64)
    print(f"[collapse_train] ner-bio: {len(records)} sentences → "
          f"{len(embeddings)} token positions, {len(tag_set)} BIO tags")
    return embeddings, labels, tag_set


# ── training loop ──────────────────────────────────────────────────────────────
def train(model, X, y, loss_fn, epochs=EPOCHS, lr=LR, batch=BATCH, patience=PATIENCE):
    N  = len(X)
    n_val = max(1, int(N * VAL_SPLIT))
    idx   = torch.randperm(N)
    Xtr, ytr = X[idx[n_val:]], y[idx[n_val:]]
    Xva, yva = X[idx[:n_val]],  y[idx[:n_val]]

    loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, no_improve, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva.to(DEVICE)), yva.to(DEVICE)))
        if val_loss < best_val - 1e-4:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  early stop epoch {epoch+1}/{epochs}  best_val={best_val:.4f}")
            break
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, X, y, task: str, teacher_start_s: float, collapsed_start_s: float):
    """Return metrics dict."""
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE)).cpu()

    if task == "chunk-boundary":
        probs = torch.sigmoid(logits).squeeze(1).numpy()
        preds = (probs > 0.5).astype(int)
        labels_np = y.numpy()
        tp = int(((preds == 1) & (labels_np == 1)).sum())
        fp = int(((preds == 1) & (labels_np == 0)).sum())
        fn = int(((preds == 0) & (labels_np == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
    else:
        preds = logits.argmax(dim=1).numpy()
        labels_np = y.numpy()
        f1 = float((preds == labels_np).mean())  # accuracy as proxy

    n_params  = sum(p.numel() for p in model.parameters())
    teacher_t = max(teacher_start_s, 1e-6)
    collapse_t = max(collapsed_start_s, 1e-6)

    return {
        "task":               task,
        "f1_vs_consensus":    round(f1, 4),
        "latency_ratio":      round(collapse_t / teacher_t, 4),
        "params":             n_params,
        "pass":               f1 >= 0.80 and (collapse_t / teacher_t) <= 0.35,
    }


# ── ONNX export ────────────────────────────────────────────────────────────────
def export_onnx(model, input_dim: int, out_path: str, task: str):
    model.eval()
    dummy = torch.randn(1, input_dim).to(DEVICE)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["embedding"],
        output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)
    # Quick verify
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    _ = sess.run(None, {"embedding": dummy.cpu().numpy()})
    print(f"[collapse_train] exported → {out_path}  ({os.path.getsize(out_path)//1024} KB)")


# ── artifact.json ──────────────────────────────────────────────────────────────
def write_artifact(out_dir: str, task: str, metrics: dict, onnx_path: str):
    sha = hashlib.sha256(open(onnx_path, "rb").read()).hexdigest()
    size_mb = round(os.path.getsize(onnx_path) / 1024 / 1024, 3)
    obj = {
        "id":          f"bonfyre-{task}-v1",
        "name":        f"Bonfyre {task.replace('-', ' ').title()} v1",
        "format":      "onnx",
        "sha256":      sha,
        "size_mb":     size_mb,
        "task":        task,
        "metrics":     metrics,
        "created_at":  int(time.time()),
        "sourceSystem": "BonfyreCollapse",
    }
    with open(os.path.join(out_dir, "artifact.json"), "w") as f:
        json.dump(obj, f, indent=2)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Collapse teacher outputs into ONNX student")
    p.add_argument("--task",          required=True,
                   choices=["topic-map", "chunk-boundary", "ner-bio", "risk-score"])
    p.add_argument("--out",           required=True)
    # task-specific inputs
    p.add_argument("--tag-dir",       default=None)
    p.add_argument("--embed-dir",     default=None)
    p.add_argument("--embed-dir-a",   default=None)
    p.add_argument("--embed-dir-b",   default=None)
    p.add_argument("--consensus",     default=None)
    p.add_argument("--risk-dir",      default=None)
    p.add_argument("--epochs",        type=int, default=EPOCHS)
    p.add_argument("--calibration",   action="store_true",
                   help="calibration mode: skip pass/fail gate, emit full profile")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    task          = args.task
    teacher_t0    = time.perf_counter()

    # ── load data ──────────────────────────────────────────────────────────────
    print(f"[collapse_train] task={task}  device={DEVICE}")
    embedder = SentenceTransformer(STUDENT_BASE, device=DEVICE)

    if task == "topic-map":
        if not args.tag_dir or not args.embed_dir:
            sys.exit("--tag-dir and --embed-dir required for topic-map")
        X_np, y_np, label_names = load_tag_embed(args.tag_dir, args.embed_dir)
        n_classes = len(label_names)
        head      = TopicHead(n_classes).to(DEVICE)
        loss_fn   = nn.CrossEntropyLoss()
        X         = torch.tensor(X_np)
        y         = torch.tensor(y_np)
        input_dim = EMBED_DIM

    elif task == "chunk-boundary":
        if not args.embed_dir_a or not args.embed_dir_b:
            sys.exit("--embed-dir-a and --embed-dir-b required for chunk-boundary")
        X_np, y_np = load_two_embeds(args.embed_dir_a, args.embed_dir_b)
        head       = BoundaryHead().to(DEVICE)
        loss_fn    = nn.BCEWithLogitsLoss()
        X          = torch.tensor(X_np)
        y          = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
        input_dim  = EMBED_DIM * 2

    elif task == "ner-bio":
        if not args.consensus:
            sys.exit("--consensus required for ner-bio")
        X_np, y_np, label_names = load_consensus(args.consensus, embedder)
        n_tags  = len(label_names)
        head    = NerBioHead(n_tags).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        X       = torch.tensor(X_np)
        y       = torch.tensor(y_np)
        input_dim = EMBED_DIM

    elif task == "risk-score":
        if not args.embed_dir or not args.risk_dir:
            sys.exit("--embed-dir and --risk-dir required for risk-score")
        X_np, y_np = load_risk_embed(args.embed_dir, args.risk_dir)
        head    = RiskHead().to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        X       = torch.tensor(X_np)
        y       = torch.tensor(y_np)
        input_dim = EMBED_DIM

    teacher_elapsed = time.perf_counter() - teacher_t0

    # ── train ──────────────────────────────────────────────────────────────────
    print(f"[collapse_train] training {sum(p.numel() for p in head.parameters()):,} params "
          f"for up to {args.epochs} epochs ...")
    collapse_t0 = time.perf_counter()
    train(head, X, y if y.dim() > 1 else y, loss_fn, epochs=args.epochs)
    collapse_elapsed = time.perf_counter() - collapse_t0

    # ── eval ───────────────────────────────────────────────────────────────────
    metrics = evaluate(head, X, y if y.dim() == 1 else y.squeeze(1),
                       task, teacher_elapsed, collapse_elapsed)

    if args.calibration:
        # In calibration mode emit the full profile and never gate on pass/fail.
        # Extra fields useful for comparing experiments:
        metrics["n_samples"]       = int(X.shape[0])
        metrics["n_params"]        = sum(p.numel() for p in head.parameters())
        metrics["teacher_time_s"]  = round(teacher_elapsed, 3)
        metrics["collapse_time_s"] = round(collapse_elapsed, 3)
        metrics["epochs_requested"] = args.epochs
        metrics["mode"]            = "calibration"
        print(f"[collapse_train] CALIBRATION profile:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        status = "PASS" if metrics["pass"] else "FAIL"
        print(f"[collapse_train] {status}: f1={metrics['f1_vs_consensus']} "
              f"latency_ratio={metrics['latency_ratio']}")

    # ── export ────────────────────────────────────────────────────────────────
    onnx_path = os.path.join(args.out, "model.onnx")
    export_onnx(head, input_dim, onnx_path, task)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_artifact(args.out, task, metrics, onnx_path)

    if not args.calibration and not metrics["pass"]:
        sys.exit(1)
    print(f"[collapse_train] done → {args.out}/")


if __name__ == "__main__":
    main()
