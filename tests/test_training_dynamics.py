import pytest
import torch
import torch.optim as optim
from network import INPUT_PLANES, POLICY_OUTPUT, ChessNet


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA"),
        ),
    ],
)
class TestTrainingDynamics:

    def test_overfitting_single_batch(self, device):
        device = torch.device(device)
        model = ChessNet(num_blocks=2, channels=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        B = 4
        x = torch.randn(B, INPUT_PLANES, 8, 8).to(device)

        target_policy = torch.zeros(B, POLICY_OUTPUT).to(device)
        target_indices = torch.randint(0, POLICY_OUTPUT, (B,)).to(device)
        target_policy.scatter_(1, target_indices.unsqueeze(1), 1.0)

        target_value = torch.randn(B).clamp(-1, 1).to(device)

        initial_loss = None

        model.train()
        for i in range(300):
            optimizer.zero_grad()
            p_logits, v_pred = model(x)

            p_log_soft = torch.nn.functional.log_softmax(p_logits, dim=1)
            loss_p = -(target_policy * p_log_soft).sum(dim=1).mean()
            loss_v = torch.nn.functional.mse_loss(v_pred.squeeze(), target_value)
            loss = loss_p + loss_v

            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()

        final_loss = loss.item()

        assert (
            final_loss < initial_loss * 0.25
        ), f"Failed to overfit: initial {initial_loss:.4f} -> final {final_loss:.4f}"

    def test_gradient_flow(self, device):
        device = torch.device(device)
        model = ChessNet(num_blocks=1, channels=16).to(device)

        x = torch.randn(2, INPUT_PLANES, 8, 8, device=device, requires_grad=True)
        p, v = model(x)

        loss = p.sum() + v.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().mean() > 0, "Input gradient is zero (backprop failed)"

        has_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad += 1

        assert has_grad > total_params * 0.5, f"Too many parameters have no gradient: {has_grad}/{total_params}"

    def test_value_bounds(self, device):
        device = torch.device(device)
        model = ChessNet(num_blocks=1, channels=16).to(device)

        x = torch.randn(10, INPUT_PLANES, 8, 8).to(device)
        with torch.no_grad():
            _, v = model(x)

        assert v.min() >= -1.0
        assert v.max() <= 1.0
