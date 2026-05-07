import torch
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from models import registry
from dataloader import create_dataloaders
from utils import true_class_from_signed_pt, confusion_matrix_3, roc_curve_binary, auc_trapz
from plotting import plot_confusion, plot_roc, plot_prob_vs_pt, plot_acceptance_vs_pt
import numpy as np

MODEL_PATHS = ["output/best_model_baseline.pth"]
CONFIG_PATHS = ["config/baseline.yml"]

device = "cuda"
outdir = Path("output")

def load_model(path: str) -> torch.nn.Module:
    return torch.load(path, map_location=device, weights_only=False)

data = {}
# for model in [m1, m2, m3]:
for model_path in MODEL_PATHS:

    run_name = Path(model_path).stem

    print(f"\n[INFO] Loading model from {model_path}...")

    model = load_model(model_path).to(device)

    model_name, model_cfg = registry.class_to_config(type(model))

    input_type = model_cfg["input_type"]

    print(f"\n[INFO] Evaluating {model_name} with input_type={input_type}...")

    run_name = Path(model_path).stem.replace("best_model_", "")
    
    for config_path in CONFIG_PATHS:
    
        print(f"[INFO] Using config: {config_path}")
    
    
        _, _, test_loader = create_dataloaders(
            config_path=config_path,
            batch_size=128,
            shuffle=False,
            input_type=input_type,
            target_type="raw",  # return raw labels for most flexibility in evaluation
            val_size=0.0,
            apply_scaling=True,
        )
    
        label_names = test_loader.dataset.label_names
    
        print(f"label_names: {label_names}")
    
        pt_idx = label_names.index("pt")
    
        true_pts, logits = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(test_loader)):
                x = x.to(device)
                out = model(x)
                logits.append(out)
                true_pts.append(y[:, pt_idx])
                if i >= 1000:  # Limit to first 100 batches for quick evaluation
                    break
    
        true_pts = np.concatenate(true_pts, axis=0)
        logits = torch.cat(logits, dim=0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_classes = np.argmax(probs, axis=1).astype(np.int8)
        true_classes = true_class_from_signed_pt(true_pts, pt_boundary=0.2)
        high_true = (true_classes == 2).astype(np.int32)
        
        config_name = Path(config_path).stem.replace("config/", "")

        data[run_name +" "+ config_name] = {
            "true_pts": true_pts,
            "logits": logits,
            "probs": probs,
            "pred_classes": pred_classes,
            "true_classes": true_classes,
            "high_true": high_true,
        }

p_high = {name: d["probs"][:, 2].astype(np.float32) for name, d in data.items()}

cm_by_model = {name: confusion_matrix_3(d["true_classes"], d["pred_classes"]) for name, d in data.items()}

rocs = {}
for name, ph in p_high.items():
    fpr, tpr = roc_curve_binary(high_true, ph)
    auc = auc_trapz(fpr, tpr)
    rocs[name] = (fpr, tpr, auc)

for name, cm in cm_by_model.items():
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    print(f"[METRIC] {name} accuracy: {acc:.6f}")
for name, (_, _, auc) in rocs.items():
    print(f"[METRIC] {name} high-vs-low AUC: {auc:.6f}")

plot_confusion(cm_by_model, outdir / "confusion_matrices.png")
plot_roc(rocs, outdir / "roc_highpt.png")
plot_prob_vs_pt(true_pts, p_high, outdir / "pt_confidence_vs_true_pt.png", nbins=40)

plot_acceptance_vs_pt(true_pts, {name: data[name]["pred_classes"] for name in data.keys()}, outdir / "acceptance_vs_true_pt.png")

print("[DONE] Wrote outputs to:", str(outdir))    
