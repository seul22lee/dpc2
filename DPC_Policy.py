import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import pickle
from pickle import dump

window = 10
 
# def DPC_quantile_loss_quantiles(model_output: torch.Tensor, target: torch.Tensor, quantiles):

#     errors = target.unsqueeze(-1) - model_output
#     quantiles_tensor = torch.tensor(quantiles, device=model_output.device)
#     losses = torch.max((quantiles_tensor - 1) * errors, quantiles_tensor * errors)
#     return losses.mean()

def DPC_loss(model_output: torch.Tensor, target: torch.Tensor, u_output: torch.Tensor, c_fut: torch.Tensor):

    model_output = model_output.median(dim=-1).values
    model_output_x1 = model_output[:, :, 0]
    target_x1 = target[:, :, 0]
    errors = (target_x1 - model_output_x1) ** 2

    u_diff = u_output[:, 1:, :] - u_output[:, :-1, :]
    squared_diff = u_diff ** 2

    model_output_x2 = model_output[:, :, 1]
    low_violation = F.relu(c_fut[:, :, 0] - model_output_x2) ** 2  # Low constraint violation
    up_violation = F.relu(model_output_x2 - c_fut[:, :, 1]) ** 2  # Up constraint violation
    constraint_loss = low_violation + up_violation  # Sum constraint violations

    tracking_loss_sqrt = torch.sqrt(errors.mean())
    smoothness_loss_sqrt = 0.1*torch.sqrt(squared_diff.mean())
    constraint_loss_sqrt = 2*torch.sqrt(constraint_loss.mean())

    print(f"Loss -> Tracking: {tracking_loss_sqrt:.4f}, Constraint: {constraint_loss_sqrt:.4f}, Smoothness: {smoothness_loss_sqrt:.4f}")

    return tracking_loss_sqrt # + smoothness_loss_sqrt + constraint_loss_sqrt 


# def DPC_PolicyNN_forward(u_hat: np.array, u_past: np.array, x_past: np.array, SP_hat: np.array, P, DPC_Policy):
#     # convert u_hat into tensor
#     u_hat = torch.tensor(u_hat.reshape(-1, 1), requires_grad=False, dtype=torch.float32)
#     u_hat_in = u_hat.unsqueeze(0)

#     # prepare past covariates
#     past_cov = torch.tensor(np.concatenate((x_past, u_past), axis=0), dtype=torch.float32).transpose(1, 0).unsqueeze(0)

#     # model prediction
#     x_hat = DPC_Policy([past_cov, u_hat_in, None])

#     # compute objective value
#     return x_hat[0, :, :, 1], x_hat

class DPC_PolicyNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_dim: int
    ):
        super(DPC_PolicyNN, self).__init__()

        self.input_dim = input_dim  
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # Fully connected layers for encoder
        self.fc1 = nn.Linear(60, hidden_dim)  # Updated input size to 60
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim * output_chunk_length)

        self.relu = nn.ReLU()

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        x, x_future_covariates, c_fut, x_static_covariates = x_in

        if len(x.shape) == 3:
            x = x.flatten(start_dim=1)  # Flatten the input tensor to match the input size

        if c_fut is not None:
            c_fut = c_fut.flatten(start_dim=1)
            x = torch.cat([x, c_fut], dim=1)

        # Concatenate future and static covariates if provided
        if x_future_covariates is not None:
            x_future_covariates = x_future_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_future_covariates], dim=1)

        if x_static_covariates is not None:
            x_static_covariates = x_static_covariates.flatten(start_dim=1)
            x = torch.cat([x, x_static_covariates], dim=1)

        # Pass through the fully connected layers with ReLU activations
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)

        # Reshape the output for the final prediction
        batch_size = x.shape[0]
        x = x.view(batch_size, self.output_chunk_length, self.output_dim, 1)

        return x
