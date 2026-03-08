from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


@dataclass
class AEConfig:
    """
    Configuration container for the baseline autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of input features in the model-ready dataset.
    hidden_dim_1 : int, default=64
        Width of the first hidden layer in the encoder/decoder.
    hidden_dim_2 : int, default=32
        Width of the second hidden layer in the encoder/decoder.
    latent_dim : int, default=16
        Size of the bottleneck representation.
    dropout : float, default=0.0
        Dropout rate applied after hidden activations. Keep at 0.0
        for a plain baseline unless you want regularization.
    """
    input_dim: int
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    latent_dim: int = 16
    dropout: float = 0.0


class BaselineAutoencoder(nn.Module):
    """
    Simple fully connected autoencoder for tabular anomaly detection.

    Architecture
    ------------
    input -> hidden_dim_1 -> hidden_dim_2 -> latent_dim
          -> hidden_dim_2 -> hidden_dim_1 -> input

    Notes
    -----
    - The encoder compresses the input into a lower-dimensional latent space.
    - The decoder reconstructs the original input from that latent space.
    - During training, the model learns to reconstruct normal samples well.
    - At inference time, large reconstruction error suggests an anomaly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        latent_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Encoder: compresses the input into a smaller latent representation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, latent_dim),
        )

        # Decoder: reconstructs the original feature vector from the latent code.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed batch with the same shape as the input.
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def reconstruction_error(
    model: nn.Module,
    X_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Compute per-row mean squared reconstruction error.

    Parameters
    ----------
    model : nn.Module
        Trained autoencoder model.
    X_tensor : torch.Tensor
        Input data tensor of shape (n_samples, n_features).
    device : torch.device
        Device used for inference ("cpu" or "cuda").

    Returns
    -------
    np.ndarray
        One reconstruction error value per row.

    Notes
    -----
    - We use the mean squared error across features for each observation.
    - Higher error means the model reconstructed that sample poorly.
    - In anomaly detection, higher error is interpreted as more anomalous.
    - Non-finite values are replaced so downstream metrics do not crash
      Quarto rendering or sklearn scoring functions.
    """

    def _safe_scores(x: np.ndarray, fill_value: float | None = None) -> np.ndarray:
        """
        Replace NaN / +/-inf values in anomaly scores with finite values.
        """
        x = np.asarray(x, dtype=np.float64)
        finite_mask = np.isfinite(x)

        # If everything is already finite, return as-is
        if finite_mask.all():
            return x

        # If no finite values exist, fall back to zeros
        if not finite_mask.any():
            print("Warning: reconstruction_error produced no finite values; returning zeros.")
            return np.zeros_like(x, dtype=np.float64)

        finite_vals = x[finite_mask]

        # Default replacement strategy:
        # - nan and +inf -> largest finite value
        # - -inf         -> smallest finite value
        if fill_value is None:
            fill_value = np.max(finite_vals)

        x_safe = np.nan_to_num(
            x,
            nan=fill_value,
            posinf=fill_value,
            neginf=np.min(finite_vals),
        )

        n_bad = np.size(x) - np.count_nonzero(finite_mask)
        print(f"Warning: replaced {n_bad} non-finite reconstruction error value(s).")

        return x_safe

    model.eval()

    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        X_hat = model(X_tensor)

        # Mean squared reconstruction error for each row
        err = torch.mean((X_tensor - X_hat) ** 2, dim=1)

    err_np = err.detach().cpu().numpy()

    # Sanitize before returning so downstream scoring won't crash
    return _safe_scores(err_np)


def tune_threshold_max_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_grid: int = 200,
) -> tuple[float, float]:
    """
    Choose the anomaly threshold that maximizes F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels where 0 = normal and 1 = attack.
    scores : np.ndarray
        Anomaly scores for each sample. Here, these are reconstruction errors.
    n_grid : int, default=200
        Number of threshold candidates to evaluate between the minimum and
        maximum score.

    Returns
    -------
    tuple[float, float]
        best_threshold : float
            Threshold that produced the highest F1 score.
        best_f1 : float
            Best validation F1 score obtained.

    Notes
    -----
    - A sample is classified as an attack when score > threshold.
    - This function is intended for validation-set threshold tuning.
    - The chosen threshold should then be applied once to the test set.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    thresholds = np.linspace(scores.min(), scores.max(), n_grid)

    best_thr = float(thresholds[0])
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = (scores > thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)

    return best_thr, best_f1

def train_autoencoder(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 30,
) -> list[float]:
    """
    Train an autoencoder on normal-only training data.

    Parameters
    ----------
    model : nn.Module
        Autoencoder model to train.
    train_loader : DataLoader
        Mini-batch loader built from normal training samples only.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    criterion : nn.Module
        Reconstruction loss function, typically MSELoss.
    device : torch.device
        Device used for training ("cpu" or "cuda").
    num_epochs : int, default=30
        Number of full passes through the training data.

    Returns
    -------
    list[float]
        Average training loss for each epoch.

    Notes
    -----
    - Each batch is reconstructed and compared against itself.
    - Lower loss means the model is better at reproducing normal patterns.
    - This function does not use labels because AE training is unsupervised.
    """
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in train_loader:
            # TensorDataset returns a tuple, so batch[0] is the feature tensor.
            x_batch = batch[0].to(device)

            optimizer.zero_grad()
            x_hat = model(x_batch)
            loss = criterion(x_hat, x_batch)
            loss.backward()
            optimizer.step()

            # Accumulate weighted batch loss so we can compute epoch average.
            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        print(f"Epoch [{epoch + 1:02d}/{num_epochs}] - train_loss: {epoch_loss:.6f}")

    return train_losses


def convert_to_fp16(model: nn.Module) -> nn.Module:
    """
    Convert a trained autoencoder to half precision (float16).

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model with float32 weights.

    Returns
    -------
    nn.Module
        Model with weights converted to float16.

    Notes
    -----
    - Float16 reduces memory usage by ~50%.
    - No retraining is required.
    - Useful for evaluating performance trade-offs in lightweight deployments.
    """
    model_fp16 = model.half()
    return model_fp16


def convert_to_uint8_dynamic(model: nn.Module) -> nn.Module:
    """
    Convert a trained autoencoder to an 8-bit dynamically quantized model.

    Parameters
    ----------
    model : nn.Module
        Trained float32 autoencoder model.

    Returns
    -------
    nn.Module
        Dynamically quantized model with Linear layers quantized to int8.

    Notes
    -----
    - This is post-training quantization for inference.
    - It is primarily intended for CPU evaluation.
    - Activations are quantized dynamically at runtime; Linear weights are
      stored in int8 form internally.
    - Input tensors should remain float32 for this model.
    """
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.cpu().eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model