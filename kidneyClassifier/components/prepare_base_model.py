from pathlib import Path
from typing import Union, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from ensure import ensure_annotations
import logging
import sys
from kidneyClassifier.exception import KidneyException
from kidneyClassifier.entity.config_entity import PrepareBaseModelConfig
from kidneyClassifier.entity.artifact_entity import PrepareBaseModelArtifact


class PrepareBaseModel:
    def __init__(
        self,
        config: dict,
        prepare_base_model_config: PrepareBaseModelConfig
    ):
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config
    
    def download_base_model(self) -> Path:
        """
        Download the base model from torchvision and save it.
        
        Returns:
            Path: Path to the saved base model file.
        """
        try:
            logging.info("Entered download_base_model")
            
            # Extract base model configuration
            base_model_config = self.config.get('base_model_config', {})
            model_name = base_model_config.get('base_model_name')
            weights_name = base_model_config.get('base_model_weights')
            
            logging.info(f"Model name: {model_name}, Weights: {weights_name}")
            
            # Create save directory
            save_dir = Path(self.prepare_base_model_config.base_model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Get the model class from torchvision.models
            model_class = getattr(models, model_name)
            
            # Get the weights class
            weights_enum = getattr(models, f"{model_name.upper()}_Weights")
            weights = getattr(weights_enum, weights_name)
            
            logging.info(f"Downloading {model_name} with {weights_name} weights...")
            
            # Download the model with pretrained weights
            model = model_class(weights=weights)
            
            # Use the file path from config
            model_save_path = Path(self.prepare_base_model_config.base_model_file_path)
            
            # Save the model
            torch.save(model.state_dict(), model_save_path)
            
            logging.info(f"Base model saved successfully at: {model_save_path}")
            
            return model_save_path
            
        except AttributeError as e:
            logging.error(f"Model or weights not found: {model_name}, {weights_name}")
            raise KidneyException(f"Invalid model name or weights: {e}", sys)
        except Exception as e:
            logging.error(f"Error occurred while downloading base model: {e}")
            raise KidneyException(e, sys)
    
    def update_base_model(
        self,
        base_model_path: Union[str, Path],
        device: Optional[str] = None
    ) -> Path:
        """
        Update the base model for custom classification task.
        
        Args:
            base_model_path: Path to the saved base model.
            device: Device to load model on ('cpu', 'cuda', or None for auto).
        
        Returns:
            Path: Path to the saved updated model file.
        """
        try:
            logging.info(f"Entered update_base_model with base_model_path={base_model_path}")
            
            # Extract configurations
            base_model_config = self.config.get('base_model_config', {})
            updated_model_config = self.config.get('updated_model_config', {})
            
            model_name = base_model_config.get('base_model_name')
            weights_name = base_model_config.get('base_model_weights')
            num_classes = updated_model_config.get('num_classes', 2)
            unfreeze_last_n_conv = updated_model_config.get('unfreeze_last_n_conv', 0)
            unfreeze_last_n_fc = updated_model_config.get('unfreeze_last_n_fc', 0)
            dropout_rate = updated_model_config.get('dropout_rate', 0.5)
            use_batch_norm = updated_model_config.get('use_batch_norm', False)
            
            logging.info(f"Configuration - num_classes: {num_classes}, "
                        f"unfreeze_last_n_conv: {unfreeze_last_n_conv}, "
                        f"unfreeze_last_n_fc: {unfreeze_last_n_fc}")
            
            # Set device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {device}")
            
            # Load the base model architecture
            model_class = getattr(models, model_name)
            model = model_class(weights=None)
            
            # Load the saved state dict
            base_model_path = Path(base_model_path)
            state_dict = torch.load(base_model_path, map_location=device)
            model.load_state_dict(state_dict)
            logging.info(f"Loaded model weights from: {base_model_path}")
            
            # Freeze all layers initially
            for param in model.parameters():
                param.requires_grad = False
            logging.info("All layers frozen initially")
            
            # Unfreeze last N convolutional layers
            if unfreeze_last_n_conv > 0:
                conv_layers = list(model.features.children())
                layers_to_unfreeze = conv_layers[-unfreeze_last_n_conv:]
                
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                logging.info(f"Unfroze last {unfreeze_last_n_conv} convolutional layer(s)")
            
            # Get the number of input features to the classifier
            num_features = model.classifier[0].in_features
            
            # Create new classifier
            if use_batch_norm:
                new_classifier = nn.Sequential(
                    nn.Linear(num_features, 4096),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(4096),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(4096),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(4096, num_classes)
                )
            else:
                new_classifier = nn.Sequential(
                    nn.Linear(num_features, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(4096, num_classes)
                )
            
            # Replace the classifier
            model.classifier = new_classifier
            logging.info(f"Replaced classifier with {num_classes} output classes")
            
            # Unfreeze last N fully connected layers
            if unfreeze_last_n_fc > 0:
                fc_layers = list(model.classifier.children())
                linear_layers = [layer for layer in fc_layers if isinstance(layer, nn.Linear)]
                layers_to_unfreeze = linear_layers[-unfreeze_last_n_fc:]
                
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                logging.info(f"Unfroze last {unfreeze_last_n_fc} fully connected layer(s)")
            else:
                for param in model.classifier.parameters():
                    param.requires_grad = True
                logging.info("Unfroze all classifier layers")
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.2f}%)")
            
            # Create save directory
            save_dir = Path(self.prepare_base_model_config.updated_model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the updated model
            updated_model_path = Path(self.prepare_base_model_config.updated_model_file_path)
            torch.save(model.state_dict(), updated_model_path)
            
            logging.info(f"Updated model saved successfully at: {updated_model_path}")
            
            # Save model architecture summary
            summary_path = Path(self.prepare_base_model_config.model_summary_path)
            with open(summary_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Weights: {weights_name}\n")
                f.write(f"Number of classes: {num_classes}\n")
                f.write(f"Dropout rate: {dropout_rate}\n")
                f.write(f"Use batch norm: {use_batch_norm}\n")
                f.write(f"Unfrozen conv layers: {unfreeze_last_n_conv}\n")
                f.write(f"Unfrozen FC layers: {unfreeze_last_n_fc}\n")
                f.write(f"Trainable parameters: {trainable_params:,} / {total_params:,}\n")
                f.write(f"\nModel Architecture:\n{model}\n")
            
            logging.info(f"Model summary saved at: {summary_path}")
            
            return updated_model_path
            
        except Exception as e:
            logging.error(f"Error occurred while updating base model: {e}")
            raise KidneyException(e, sys)
    
    def initiate_base_model(self) -> PrepareBaseModelArtifact:
        """
        Main orchestrator function that downloads and updates the base model.
        
        Returns:
            PrepareBaseModelArtifact: Artifact containing paths to base and updated models
        """
        try:
            logging.info("=" * 80)
            logging.info("INITIATING BASE MODEL PREPARATION")
            logging.info("=" * 80)
            
            # Step 1: Download base model
            logging.info("Step 1/2: Downloading base model...")
            base_model_path = self.download_base_model()
            logging.info(f"✓ Base model downloaded and saved at: {base_model_path}")
            
            # Step 2: Update base model for classification
            logging.info("Step 2/2: Updating base model for classification...")
            updated_model_path = self.update_base_model(
                base_model_path=base_model_path
            )
            logging.info(f"Model updated and saved at: {updated_model_path}")
            
            logging.info("=" * 80)
            logging.info("BASE MODEL PREPARATION COMPLETED SUCCESSFULLY")
            logging.info("=" * 80)
            
            # Create artifact
            prepare_base_model_artifact = PrepareBaseModelArtifact(
                base_model_path=str(base_model_path),
                updated_model_path=str(updated_model_path),
                model_summary_path=str(self.prepare_base_model_config.model_summary_path),
                is_artifact_valid=True
            )
            
            return prepare_base_model_artifact
            
        except Exception as e:
            logging.error("=" * 80)
            logging.error("BASE MODEL PREPARATION FAILED")
            logging.error("=" * 80)
            raise KidneyException(e, sys)