import os
import time
import torch
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm.auto import tqdm
from torch.nn.functional import mse_loss, smooth_l1_loss, cross_entropy
from torch.utils.data import DataLoader

from src.utils.config import Config
from src.utils.drive import get_drive
from src.utils.tpu import get_tpu
from src.utils.chess import H5Dataset, get_move_count
from src.utils.train import set_seed, get_device
from src.model import ChessModel

class EvalPipeline:
    def __init__(self, config: Config):
        self.config = config
        
        seed = config.get('project.seed', 42)
        set_seed(seed)
        
        self.device_info = get_device()
        self.device = self.device_info["device"]
        self.device_type = self.device_info["type"]
        
        self.use_tpu = self.device_type == "tpu"
        print(f"Using device: {self.device_type}")
        
        self.max_samples = config.get('eval.max_samples', 10000)
        
        self.model_path = None
        self.model = None
        
        self.dataset = config.get('data.dataset', 'data/dataset.h5')
        self.test_idx = config.get('data.test_idx', 'data/test_indices.npy')
        
        self.output_dir = '/content/drive/MyDrive/chess_ai/evaluation'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup(self):
        print("Setting up evaluation pipeline...")
        
        model_paths = [
            ('/content/drive/MyDrive/chess_ai/models/reinforcement_model.pth', 'Reinforcement'),
            ('/content/drive/MyDrive/chess_ai/models/supervised_model.pth', 'Supervised')
        ]
        
        drive = get_drive()
        
        for path, model_type in model_paths:
            try:
                local_path = f'/content/{path}'
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    self.model_path = drive.load(path, local_path)
                    print(f"Loaded {model_type} model from Drive: {self.model_path}")
                    break
                except FileNotFoundError:
                    if os.path.exists(local_path):
                        self.model_path = local_path
                        print(f"Using local {model_type} model: {self.model_path}")
                        break
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
        
        if not self.model_path:
            print("No model found for evaluation")
            return False
        
        try:
            local_dataset = '/content/drive/MyDrive/chess_ai/data/dataset.h5'
            local_test_idx = '/content/drive/MyDrive/chess_ai/data/test_indices.npy'
            
            os.makedirs('/content/drive/MyDrive/chess_ai/data', exist_ok=True)
            
            try:
                self.dataset = drive.load(self.dataset, local_dataset)
                self.test_idx = drive.load(self.test_idx, local_test_idx)
            except FileNotFoundError:
                print("Using original data paths")
                
            print(f"Using dataset: {self.dataset}")
            print(f"Using test indices: {self.test_idx}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            
        if self.config.get('wandb.enabled', True):
            try:
                wandb.init(
                    project=self.config.get('wandb.project', 'chess_ai'),
                    name=f"eval_{self.config.mode}_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "mode": self.config.mode,
                        "model_path": self.model_path,
                        "dataset": self.dataset,
                        "device": self.device_type,
                        "max_samples": self.max_samples
                    }
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                
        return True
    
    def load_model(self):
        if not os.path.isfile(self.model_path):
            print(f"Model file not found: {self.model_path}")
            return None
            
        try:
            ch = self.config.get('model.channels', 64)
            
            model = ChessModel(
                moves=get_move_count(),
                ch=ch,
                use_tpu=self.use_tpu
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def prepare_data(self):
        if not os.path.isfile(self.test_idx) or not os.path.isfile(self.dataset):
            print(f"Test indices or dataset file not found")
            return None
            
        try:
            test_indices = np.load(self.test_idx)
            
            if self.max_samples > 0 and len(test_indices) > self.max_samples:
                np.random.shuffle(test_indices)
                test_indices = test_indices[:self.max_samples]
            
            test_dataset = H5Dataset(self.dataset, test_indices)
            
            batch = self.config.get('data.batch', 128)
            workers = self.config.get('hardware.workers', 2)
            pin_memory = self.config.get('hardware.pin_memory', True) and not self.use_tpu
            prefetch = self.config.get('hardware.prefetch', 2)
            
            dataloader = DataLoader(
                test_dataset,
                batch_size=batch,
                shuffle=False,
                num_workers=workers,
                pin_memory=pin_memory,
                persistent_workers=(workers > 0),
                prefetch_factor=prefetch if workers > 0 else None
            )
            
            print(f"Loaded test dataset with {len(test_indices)} samples")
            return dataloader
        except Exception as e:
            print(f"Error preparing dataloader: {e}")
            return None
    
    def run_inference(self, model, dataloader):
        predictions, actuals, accuracies, logits = [], [], [], []
        
        total_batches = len(dataloader)
        print(f"Running inference on {total_batches} batches...")
        
        if self.use_tpu:
            tpu = get_tpu()
            dataloader = tpu.wrap_loader(dataloader)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, values) in enumerate(tqdm(dataloader)):
                if not self.use_tpu:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                policy_output = outputs[0]
                
                pred = policy_output.argmax(dim=1)
                accuracy = (pred == targets).float().mean().item()
                
                predictions.extend(pred.cpu().numpy().tolist())
                actuals.extend(targets.cpu().numpy().tolist())
                accuracies.append(accuracy)
                logits.append(policy_output.cpu())
                
                if wandb.run is not None and (batch_idx + 1) % 10 == 0:
                    wandb.log({
                        'batch_idx': batch_idx + 1,
                        'batch_accuracy': accuracy
                    })
        
        logits_array = torch.cat(logits, 0).numpy() if logits else np.empty((0,))
        
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        torch.cuda.empty_cache()
        
        return predictions_array, actuals_array, accuracies, logits_array
    
    def create_confusion_matrix(self, cm, classes):
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True',
            xlabel='Predicted',
            title='Confusion Matrix'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        threshold = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black"
                )
        
        fig.tight_layout()
        return fig
    
    def eval_metrics(self, predictions, actuals, accuracies):
        overall_accuracy = float(np.mean(predictions == actuals))
        avg_batch_accuracy = float(np.mean(accuracies))
        
        scores = {
            'accuracy': overall_accuracy,
            'batch_accuracy': avg_batch_accuracy
        }
        
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Average batch accuracy: {avg_batch_accuracy:.4f}")
        
        if wandb.run is not None:
            wandb.log(scores)
            wandb.summary.update(scores)
        
        unique_classes = np.unique(np.concatenate([actuals, predictions]))
        if len(unique_classes) <= 20:
            cm = metrics.confusion_matrix(actuals, predictions, labels=unique_classes)
            
            cm_figure = self.create_confusion_matrix(cm, [str(c) for c in unique_classes])
            
            if wandb.run is not None:
                wandb.log({'confusion_matrix': wandb.Image(cm_figure)})
            
            class_accuracy = cm.diagonal() / cm.sum(axis=1, where=cm.sum(axis=1) != 0)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(unique_classes)), class_accuracy)
            ax.set(
                xlabel='Class',
                ylabel='Accuracy',
                title='Per-class Accuracy',
                xticks=range(len(unique_classes)),
                xticklabels=[str(c) for c in unique_classes]
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            
            if wandb.run is not None:
                wandb.log({'per_class_accuracy': wandb.Image(fig)})
            
            cm_figure.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
            fig.savefig(os.path.join(self.output_dir, 'per_class_accuracy.png'))
            
            plt.close(cm_figure)
            plt.close(fig)
        
        if len(unique_classes) == 2:
            y_true = (actuals == unique_classes[1]).astype(int)
            y_score = (predictions == unique_classes[1]).astype(int)
            
            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
            avg_precision = metrics.average_precision_score(y_true, y_score)
            
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set(
                xlabel='Recall',
                ylabel='Precision',
                title=f'Precision-Recall Curve (AP={avg_precision:.3f})'
            )
            
            if wandb.run is not None:
                wandb.log({'precision_recall_curve': wandb.Image(fig)})
            
            fig.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'))
            plt.close(fig)
            
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            auc = metrics.auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], '--')
            ax.set(
                xlabel='False Positive Rate',
                ylabel='True Positive Rate',
                title=f'ROC Curve (AUC={auc:.3f})'
            )
            
            if wandb.run is not None:
                wandb.log({'roc_curve': wandb.Image(fig)})
            
            fig.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
            plt.close(fig)
        
        if len(actuals) > 2:
            sample_size = min(self.max_samples, len(actuals))
            idx = random.sample(range(len(actuals)), sample_size)
            
            if wandb.run is not None:
                table = wandb.Table(data=list(zip(actuals[idx], predictions[idx])), columns=['Actual', 'Predicted'])
                wandb.log({'scatter_plot': wandb.plot.scatter(table, 'Actual', 'Predicted', title='Actual vs Predicted')})
        
        return scores
    
    def analyze_gradients(self, model, dataloader, num_samples=1):
        try:
            print("Analyzing gradient-based feature importance...")
            
            iterator = iter(dataloader)
            
            for _ in range(num_samples):
                inputs, targets, _ = next(iterator)
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                inputs.requires_grad_(True)
                
                outputs = model(inputs)
                loss = cross_entropy(outputs[0], targets.long())
                loss.backward()
                
                gradient = inputs.grad.abs().mean(0).cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(range(gradient.shape[-1]), gradient.mean(0))
                ax.set(
                    xlabel='Feature Index',
                    ylabel='|∂L/∂x|',
                    title='Gradient-based Feature Importance'
                )
                fig.tight_layout()
                
                if wandb.run is not None:
                    wandb.log({'gradient_importance': wandb.Image(fig)})
                
                fig.savefig(os.path.join(self.output_dir, 'gradient_importance.png'))
                plt.close(fig)
                
                inputs.grad = None
        except Exception as e:
            print(f"Error in gradient analysis: {e}")
    
    def analyze_knockout(self, model, dataloader, knockout_idx=0):
        try:
            print(f"Analyzing feature knockout (index {knockout_idx})...")
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets, _ in dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if knockout_idx < inputs.size(1):
                        inputs[:, knockout_idx] = 0
                    
                    outputs = model(inputs)
                    predictions = outputs[0].argmax(1)
                    
                    correct += (predictions == targets).float().sum().item()
                    total += targets.size(0)
            
            accuracy = correct / total if total > 0 else 0
            print(f"Knockout accuracy (feature {knockout_idx}): {accuracy:.4f}")
            
            if wandb.run is not None:
                wandb.log({
                    'knockout_feature_idx': knockout_idx,
                    'knockout_accuracy': accuracy
                })
        except Exception as e:
            print(f"Error in feature knockout analysis: {e}")
    
    def analyze_error(self, logits, actuals):
        try:
            print("Analyzing error metrics...")
            
            if logits.size == 0 or len(actuals) == 0:
                return
            
            correct_scores = logits[np.arange(len(actuals)), actuals]
            
            mse_val = float(mse_loss(torch.tensor(correct_scores), torch.zeros_like(torch.tensor(correct_scores))))
            huber_val = float(smooth_l1_loss(torch.tensor(correct_scores), torch.zeros_like(torch.tensor(correct_scores))))
            
            print(f"MSE: {mse_val:.4f}, Huber: {huber_val:.4f}")
            
            if wandb.run is not None:
                wandb.log({
                    'mse': mse_val,
                    'huber': huber_val
                })
        except Exception as e:
            print(f"Error in error metrics analysis: {e}")
    
    def analyze_edge(self, model):
        try:
            print("Analyzing edge cases...")
            
            edge_input = torch.zeros((1, 25, 8, 8), device=self.device)
            edge_input[0, 0, 0, 0] = 1
            
            with torch.no_grad():
                outputs = model(edge_input)
                prediction = outputs[0].argmax(1).item()
            
            print(f"Edge case prediction: {prediction}")
            
            if wandb.run is not None:
                wandb.log({'edge_prediction': prediction})
        except Exception as e:
            print(f"Error in edge case analysis: {e}")
    
    def run(self):
        if not self.setup():
            return False
        
        try:
            model = self.load_model()
            if model is None:
                return False
            
            dataloader = self.prepare_data()
            if dataloader is None:
                return False
            
            predictions, actuals, accuracies, logits = self.run_inference(model, dataloader)
            
            if len(predictions) == 0:
                print("No predictions generated")
                return False
            
            self.eval_metrics(predictions, actuals, accuracies)
            
            self.analyze_gradients(model, dataloader)
            self.analyze_knockout(model, dataloader)
            self.analyze_error(logits, actuals)
            self.analyze_edge(model)
            
            if wandb.run is not None:
                wandb.finish()
            
            print("Evaluation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if wandb.run is not None:
                wandb.finish()
            return False