import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.model import ChessModel

if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/saved_models/pre_trained_model.pth"))
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    policy_shape = checkpoint.get("policy_head.4.weight", torch.empty(8064)).shape
    model = ChessModel(num_moves=policy_shape[0])
    model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 20, 8, 8)
    onnx_path = model_path.replace(".pth", ".onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_output", "value_output"],
        dynamic_axes={"input": {0: "batch_size"}, "policy_output": {0: "batch_size"}, "value_output": {0: "batch_size"}},
    )