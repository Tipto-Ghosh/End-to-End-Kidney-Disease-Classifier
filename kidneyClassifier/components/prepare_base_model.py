import os 
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from kidneyClassifier.logger import logging
from kidneyClassifier.exception import KidneyException
from kidneyClassifier.entity.config_entity import PrepareBaseModelConfig
from  kidneyClassifier.entity.artifact_entity import PrepareBaseModelArtifact


class PrepareBaseModel:
    def __init__(
        self, config: dict, prepare_base_model_config = PrepareBaseModelConfig()):
        """This class is responsible to build download the base model and
           modify the base model according to our problem statement.

        Args:
            config (dict): Configuration dictionary containing base_model_config with
                      'base_model_name' and 'base_model_weights'.
            prepare_base_model_config: 
        """
        self.config = config
        self.prepare_base_model_config = prepare_base_model_config
    
    def download_base_model(self) -> str:
        """
        Download a base model from torchvision and save it to the specified directory.

        Returns:
            Path: Path to the saved model file.

        Raises:
            KidneyException: If downloading or saving the model fails.
        """
        logging.info(f"Entered download_base_model")
        try:
            # extract base model configuration
            base_model_config = self.config.get('base_model_config' , {})
            model_name = base_model_config.get('base_model_name')
            weights_name = base_model_config.get('base_model_weights')
            
            logging.info(f"Model name: {model_name}, Weights: {weights_name}")
            
            save_dir = Path(self.prepare_base_model_config.root_dir)
            save_dir.mkdir(exist_ok = True , parents = True)
            
            # get the model class
            model_class = getattr(models , model_name)
            # Get the weights class
            weights_enum = getattr(models, f"{model_name.upper()}_Weights")
            weights = getattr(weights_enum, weights_name)
            
            # Download the model with pretrained weights
            logging.info(f"Downloading {model_name} with {weights_name} weights...")
            model = model_class(weights = weights)
            # save the model
            torch.save(
                model.state_dict(),
                self.prepare_base_model_config.base_model_file_path
            )
            logging.info(
                f"Base model saved successfully at: {self.prepare_base_model_config.base_model_file_path}"
            )
            return self.prepare_base_model_config.base_model_file_path
        
        except AttributeError as e:
            logging.error(f"Model or weights not found: {model_name}, {weights_name}")
            raise KidneyException(f"Invalid model name or weights: {e}", sys)
        
        except Exception as e:
            raise KidneyException(e , sys)
    
    def update_model_for_classification(self):
        """
        Update a base model for custom classification task by modifying the classifier
        and freezing/unfreezing layers based on configuration.

        Returns:
            Path: Path to the saved updated model file.

        Raises:
            KidneyException: If updating or saving the model fails.
        """
        
        logging.info(f"Entered update_model_for_classification")
        
        try:
            logging.info(f"Entered update_base_model")
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {device}")
            
            # Load the base model architecture
            model_class = getattr(models, model_name)
            # dont need to weights again
            model = model_class(weights = None) 
            
            # Load the saved state dict
            base_model_path = Path(
                self.prepare_base_model_config.base_model_file_path
            )
            state_dict = torch.load(base_model_path, map_location = device)
            model.load_state_dict(state_dict)
            logging.info(f"Loaded model weights from: {base_model_path}")
            
            # Freeze all layers initially
            for param in model.parameters():
                param.requires_grad = False
            logging.info("All layers frozen initially")
            
            # Unfreeze last N convolutional layers
            if unfreeze_last_n_conv > 0:
                conv_layers = list(model.features.children())
                layers_to_unfreeze = conv_layers[-unfreeze_last_n_conv : ]
                
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
                    nn.ReLU(inplace = True),
                    nn.BatchNorm1d(4096),
                    nn.Dropout(p = dropout_rate),
                    nn.Linear(4096 , 4096),
                    nn.ReLU(inplace =True),
                    nn.BatchNorm1d(4096),
                    nn.Dropout(p = dropout_rate),
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
                layers_to_unfreeze = linear_layers[-unfreeze_last_n_fc : ]
                
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                logging.info(f"Unfroze last {unfreeze_last_n_fc} fully connected layer(s)")
            else:
                # Unfreeze all classifier layers
                for param in model.classifier.parameters():
                    param.requires_grad = True
                logging.info("Unfroze all classifier layers")
            
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.2f}%)")
            
            
            # Save the updated model
            torch.save(
                model.state_dict(), 
                self.prepare_base_model_config.updated_base_model_path
            )
            
            logging.info(
            f"Updated model saved successfully at: {self.prepare_base_model_config.updated_base_model_path}"
            )
            
            # Save model architecture summary
            summary_path = self.prepare_base_model_config.updated_model_architecture_summary_file_path
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
            
            return self.prepare_base_model_config.updated_base_model_path
            
        except Exception as e:
            logging.error(f"Error occurred while updating model: {e}")
            raise KidneyException(e, sys)
    
    def initiate_base_model(self) -> PrepareBaseModelArtifact:
        """
        Main orchestrator function that downloads the base model and then updates it
        for the classification task.
        
        This function:
        1. Calls download_base_model to download and save the pretrained model
        2. Calls update_base_model with the downloaded model path to create the 
        updated model for classification
        
        Args:
            config (dict): Configuration dictionary containing both base_model_config 
                        and updated_model_config.
            base_model_dir (Union[str, Path]): Directory to save the base model.
            updated_model_dir (Union[str, Path]): Directory to save the updated model.
            device (Optional[str]): Device to load model on ('cpu', 'cuda', or None for auto).
        
        Returns:
            PrepareBaseModelArtifact(obj): returns prepare base model artifact's object.
        
        Raises:
            KidneyException: If any step in the model preparation fails.
        """
        logging.info("="*80)
        logging.info("INITIATING BASE MODEL PREPARATION")
        logging.info("="*80)
        
        try:
            # Step 1: Download base model
            logging.info("Step 1/2: Downloading base model...")
            base_model_path = self.download_base_model()
            logging.info(f"Base model downloaded and saved at: {base_model_path}")
            # Step 2: Update base model for classification
            logging.info("Step 2/2: Updating base model for classification...")
            updated_model_path = self.update_model_for_classification()
            logging.info(f"Model updated and saved at: {updated_model_path}")
            logging.info("="*80)
            logging.info("BASE MODEL PREPARATION COMPLETED SUCCESSFULLY")
            logging.info("="*80)
            
            prepare_base_model_artifact = PrepareBaseModelArtifact(
                base_model_file_path = self.prepare_base_model_config.base_model_file_path,
                updated_base_model_file_path = self.prepare_base_model_config.updated_base_model_path,
                updated_model_architecture_summary_file_path = self.prepare_base_model_config.updated_model_architecture_summary_file_path
            )
            return prepare_base_model_artifact
        except Exception as e:
            raise KidneyException(e , sys)