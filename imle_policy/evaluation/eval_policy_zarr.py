from tqdm.auto import tqdm
import torch
import numpy as np

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

    with torch.no_grad():
        with tqdm(dataloader, desc='Eval Batch', leave=False) as tepoch:
            for batch in tepoch:
                obs = batch['obs'].to(device)
                B, H, D = obs.shape
                
                obs = obs.reshape(B, H * D)
                target_dim = nets['policy_net'].module.global_cond_dim
                if obs.shape[1] != target_dim:
                    assert target_dim % obs.shape[1] == 0, (
                        f"Cannot Expand obs {obs.shape[1]} -> {target_dim}"
                    )
                    factor = target_dim // obs.shape[1]
                    obs = obs.repeat(1, factor)
                true_action = batch['action'].to(device)

                if method == 'rs_imle':
                    noise = torch.randn(obs.shape[0], *true_action.shape[1:], device=device)
                    pred_action = nets['policy_net'](obs, noise)
                else:
                    # for diffusion or other methods, fallback to using obs as input
                    pred_action = nets['policy_net'](obs, torch.randn_like(true_action))

                distances = torch.linalg.norm(pred_action - true_action, dim=2)
                number_of_elements = torch.numel(distances)
                mse = torch.mean((pred_action - true_action) ** 2).item()
                
                total_mse += mse * distances.shape[0]
                total_distance += distances.sum().item()
                total_first_action_distance += distances[:, 0].sum().item()

                total_samples += distances.shape[0]
                total_actions += number_of_elements

                tepoch.set_postfix(mse=mse)

    mean_mse = total_mse / total_samples
    mean_distance = total_distance / total_actions
    mean_first_action_distance = total_first_action_distance / total_samples

    return mean_mse, mean_distance, mean_first_action_distance