import torch
from 1_load_merge import load_model

def export():
    model, tok = load_model()
    model.eval()

    dummy = tok("hello world", return_tensors="pt")

    exported = torch.export.export(
        model,
        (dummy["input_ids"],),
        strict=False
    )

    exported.save("model.pt2")
    print("graph saved")

if __name__ == "__main__":
    export()
