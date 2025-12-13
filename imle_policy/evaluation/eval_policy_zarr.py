from tqdm.auto import tqdm
import torch
import numpy as np
from utils.losses import rs_imle_loss

def get_eval_metrics(nets, batch, method, args_dict, device):
    obs = batch['obs'].to(device)
    B, H, D = obs.shape
    
    obs = obs.reshape(B, H * D)
    true_action = batch['action'].to(device)

    if method == 'rs_imle' and args_dict["architecture"] == "unet":
        noise = torch.randn(obs.shape[0], *true_action.shape[1:], device=device)
        pred_action = nets['policy_net'](obs, noise)
    elif method == 'rs_imle' and args_dict["architecture"] == "transformer":
        noise = torch.randn(obs.shape[0], args_dict["noise_dim"], device=device)
        pred_action = nets['policy_net'](obs, noise)
        pred_action = pred_action.reshape(obs.shape[0], args_dict["pred_horizon"], args_dict["action_dim"])
    else:
        # for diffusion or other methods, fallback to using obs as input
        pred_action = nets['policy_net'](obs, torch.randn_like(true_action))

    distances = torch.linalg.norm(pred_action - true_action, dim=2)
    number_of_elements = torch.numel(distances)
    mse = torch.mean((pred_action - true_action) ** 2).item()

    return (
        mse * distances.shape[0],
        distances.sum().item(),
        distances[:, 0].sum().item(),
        distances.shape[0],
        number_of_elements,
    )

def get_loss_metrics(batch, nets, args_dict, device):
    from imle_policy.train import train_rs_imle_unet_step, train_rs_imle_transformer_step
    
    obs = batch['obs'].to(device)
    B, H, D = obs.shape
    
    obs = obs.reshape(B, H * D)
    true_action = batch['action'].to(device)

    if args_dict["architecture"] == "transformer":
        return train_rs_imle_transformer_step(nets, obs, true_action, B, args_dict, device)
    else:
        return train_rs_imle_unet_step(nets, obs, true_action, B, args_dict, device)

def evaluate(args_dict, nets, stats, method):
    """
    Evaluate the policy network on your Zarr dataset.

    Returns:
        mean_cov: Mean squared error (MSE) across dataset
        mean_success: Placeholder (can be used for other metrics, here just 0)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets.eval()

    # Use the same dataset as during training
    from imle_policy.dataloaders.dataset_zarr import PolicyDataset
    dataset = PolicyDataset(
        dataset_path=args_dict['dataset_path'],
        pred_horizon=args_dict['pred_horizon'],
        obs_horizon=args_dict['obs_horizon'],
        action_horizon=args_dict['action_horizon'],
        dataset_percentage=(0.9, 1.0)  # sample 10% for quick evaluation
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args_dict['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    total_mse = 0.0
    total_distance = 0.0
    total_first_action_distance = 0.0
    total_samples = 0.0
    total_actions = 0.0

    loss_max_distance = 0.0
    loss_min_distance = 0.0
    loss_mean_distance = 0.0
    batch_count = 0

    with torch.no_grad():
        with tqdm(dataloader, desc='Eval Batch', leave=False) as tepoch:
            for batch in tepoch:

                (mse, distance, first_action_distance, samples, actions) = get_eval_metrics(nets, batch, method, args_dict, device)
                total_mse += mse
                total_distance += distance
                total_first_action_distance += first_action_distance

                total_samples += samples
                total_actions += actions

                _, wandb_log = get_loss_metrics(batch, nets, args_dict, device)
                loss_max_distance += wandb_log["max_distance"]
                loss_min_distance += wandb_log["min_distance"]
                loss_mean_distance += wandb_log["mean_distance"]
                batch_count += 1

                tepoch.set_postfix(mse=mse)

    mean_mse = total_mse / total_samples
    mean_distance = total_distance / total_actions
    mean_first_action_distance = total_first_action_distance / total_samples

    loss_max_distance = loss_max_distance / batch_count
    loss_min_distance = loss_min_distance / batch_count
    loss_mean_distance = loss_mean_distance / batch_count

    return mean_mse, mean_distance, mean_first_action_distance,\
        loss_max_distance, loss_min_distance, loss_mean_distance