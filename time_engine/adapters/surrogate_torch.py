try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

class SurrogateAdapterTorch:
    """Tiny MLP surrogate for local-grid one-step prediction."""
    def __init__(self, params=None, model_path=None, device="cpu"):
        if torch is None:
            raise RuntimeError("PyTorch not available for SurrogateAdapterTorch")
        self.params = params or {}
        self.device = device
        input_size = int(self.params.get("input_size", 16))
        hidden = int(self.params.get("hidden", 64))
        self.model = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Linear(hidden, input_size)).to(self.device)
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception:
                pass

    def predict(self, local_state, dt):
        import numpy as _np
        arr = _np.asarray(local_state, dtype=float)
        # pad or trim to model input_size
        inp = arr.flatten()
        sz = self.model[0].in_features
        if inp.size < sz:
            inp = _np.pad(inp, (0, sz - inp.size), 'constant')
        else:
            inp = inp[:sz]
        x = torch.tensor(inp, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x).squeeze(0).cpu().numpy()
        # map back to same shape as input (approx)
        return out.tolist()


