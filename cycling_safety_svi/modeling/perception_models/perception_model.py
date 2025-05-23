import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification, ConvNextV2Model
import timm  # Import timm for RepVit and EfficientViT models

class CrossAttention(nn.Module):
    """
    Cross-attention mechanism to allow different perception heads to attend to each other's features
    """
    def __init__(self, feature_dim, num_heads=4):
        """
        Initialize the cross-attention module
        
        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
        """
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Ensure feature_dim is divisible by num_heads
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, features_list):
        """
        Apply cross-attention to a list of features
        
        Args:
            features_list (list): List of feature tensors [batch_size, feature_dim]
            
        Returns:
            List of attended feature tensors [batch_size, feature_dim]
        """
        batch_size = features_list[0].shape[0]
        head_dim = self.feature_dim // self.num_heads
        
        # Process each feature tensor with cross-attention
        attended_features = []
        
        for i, query_features in enumerate(features_list):
            # Apply layer normalization
            query_features = self.norm1(query_features)
            
            # Project to queries, keys, and values
            queries = self.q_proj(query_features).view(batch_size, self.num_heads, head_dim)
            
            # Initialize attention outputs
            attention_output = torch.zeros_like(query_features)
            
            # Attend to features from other heads
            for j, key_features in enumerate(features_list):
                if i != j:  # Only attend to other features, not self
                    # Project to keys and values
                    keys = self.k_proj(key_features).view(batch_size, self.num_heads, head_dim)
                    values = self.v_proj(key_features).view(batch_size, self.num_heads, head_dim)
                    
                    # Compute attention scores
                    scores = torch.matmul(queries.unsqueeze(3), keys.unsqueeze(2))
                    scores = scores / (head_dim ** 0.5)  # Scale by sqrt(head_dim)
                    
                    # Apply softmax
                    attention_weights = torch.softmax(scores, dim=-1)
                    
                    # Apply attention weights to values
                    attended = torch.matmul(attention_weights, values.unsqueeze(3)).squeeze(3)
                    
                    # Reshape and project back
                    attended = attended.view(batch_size, self.feature_dim)
                    
                    # Add to output
                    attention_output = attention_output + attended
            
            # Residual connection
            attn_output = query_features + self.out_proj(attention_output)
            
            # Apply feed-forward network with residual connection
            output = self.norm2(attn_output)
            output = attn_output + self.ffn(output)
            
            attended_features.append(output)
        
        return attended_features


class PerceptionModel(nn.Module):
    """
    Model for predicting perception variables (traffic safety, social safety, beautiful) from images
    Based on the cvdcm model architecture but adapted for direct perception prediction
    """

    def __init__(self, num_classes=3, ordinal_levels=5, hidden_layers=2, 
                 hidden_dims=[512, 256], dropout_rates=[0.3, 0.2], 
                 use_cross_attention=False, cross_attn_heads=None,
                 freeze_backbone=False, path_pretrained_model=None,
                 backbone_dropout=0.0, stochastic_depth_rate=0.0,
                 model_type="deit_base"):
        """
        Initialize the perception model
        
        Args:
            num_classes (int): Number of perception variables to predict (default: 3 - traffic safety, social safety, beautiful)
            ordinal_levels (int): Number of ordinal levels for each perception variable (default: 5)
            hidden_layers (int): Number of hidden layers in each perception head
            hidden_dims (list): Dimensions of hidden layers
            dropout_rates (list): Dropout rates for each hidden layer
            use_cross_attention (bool): No longer used - kept for backwards compatibility
            cross_attn_heads (int): No longer used - kept for backwards compatibility
            freeze_backbone (bool): Whether to freeze the vision model backbone
            path_pretrained_model (str): Path to a pretrained model to load weights from
            backbone_dropout (float): Dropout rate to apply to backbone features
            stochastic_depth_rate (float): Stochastic depth rate for additional regularization
            model_type (str): Type of model to use as backbone ("deit_base", "convnextv2_tiny", "repvit_m2_3", or "efficientvit_m1")
        """
        super(PerceptionModel, self).__init__()
        
        # Define the CNN backbone based on model_type
        self.model_type = model_type
        
        if model_type == "convnextv2_tiny":
            # ConvNextV2 model that returns features instead of classification logits
            self.vision_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-22k-224")
            self.feature_dim = 1000  # Output dimension matches DeiT model
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Add a feature projection layer to map from backbone feature dimension to our model's feature dimension
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            
            # Add batch normalization and activation for the projected features
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        elif model_type == "repvit_m2_3":
            # RepVit model from timm
            base_model = timm.create_model('repvit_m2_3.dist_450e_in1k', pretrained=True)
            
            # Extract the feature dimension from the classifier
            # For RepVit models, the classifier is a Classfier with an embedding_size in the source code
            # Looking at the source code, we see it's the last layer's output dimension, which is 640 for m2_3
            self.feature_dim = 640  # Fixed dimension for repvit_m2_3 based on source code
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Create a new model without the classifier head
            self.vision_model = timm.create_model(
                'repvit_m2_3.dist_450e_in1k', 
                pretrained=True,
                num_classes=0  # This removes the classifier head
            )
            
            # Add a feature projection layer
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        elif model_type == "efficientvit_m1":
            # EfficientViT model from timm
            # Create a model that returns features only (no classification head)
            self.vision_model = timm.create_model(
                'efficientvit_m1.r224_in1k',
                pretrained=True,
                num_classes=0  # Remove classifier head
            )
            
            # For EfficientViT M1, the feature dimension is 192
            self.feature_dim = 192
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Add a feature projection layer
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        else:
            # Default DeiT model
            self.vision_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=True)
            self.feature_dim = 1000  # Output dimension of the model
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # For consistency, also add projection for DeiT
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        
        # Freeze the vision model if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Ensure hidden_dims and dropout_rates have the correct length
        if len(hidden_dims) < hidden_layers:
            hidden_dims = hidden_dims + [hidden_dims[-1]] * (hidden_layers - len(hidden_dims))
        if len(dropout_rates) < hidden_layers:
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (hidden_layers - len(dropout_rates))
            
        # Truncate if longer than needed
        hidden_dims = hidden_dims[:hidden_layers]
        dropout_rates = dropout_rates[:hidden_layers]
        
        # Store architecture parameters
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.use_cross_attention = False  # Always set to False
        
        # Add input dropout to prevent overfitting from the start
        self.input_dropout = nn.Dropout(backbone_dropout if backbone_dropout > 0 else 0.2)
        
        # Add stochastic depth if specified
        self.use_stochastic_depth = stochastic_depth_rate > 0
        if self.use_stochastic_depth:
            self.stochastic_depth = stochastic_depth_rate
        
        # Define the feature extraction heads (separate for each perception variable)
        self.feature_extractors = nn.ModuleList()
        
        # Create feature extraction heads
        for _ in range(num_classes):
            layers = []
            input_dim = self.projected_dim  # Use projected dimension
            
            # Add hidden layers except the last one
            for i in range(hidden_layers - 1):
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
                layers.append(nn.BatchNorm1d(hidden_dims[i]))  # Add batch normalization
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rates[i]))
                input_dim = hidden_dims[i]
            
            # Create sequential model for this head
            self.feature_extractors.append(nn.Sequential(*layers))
        
        # Final classification heads (separate for each perception variable)
        self.classifiers = nn.ModuleList()
        
        # Create final classification layers
        for _ in range(num_classes):
            layers = []
            
            # Add only the final layer
            # If only one hidden layer, use the output of vision model
            if hidden_layers > 1:
                input_dim = hidden_dims[-2]  # Use the second-to-last hidden dim
            else:
                input_dim = self.projected_dim  # Use the projected dimension
                
            layers.append(nn.Linear(input_dim, hidden_dims[-1]))
            layers.append(nn.BatchNorm1d(hidden_dims[-1]))  # Add batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[-1]))
            layers.append(nn.Linear(hidden_dims[-1], ordinal_levels))
            
            # Create sequential model for this head
            self.classifiers.append(nn.Sequential(*layers))
        
        # Load pretrained model if provided
        if path_pretrained_model is not None:
            self.load_state_dict(torch.load(path_pretrained_model), strict=True)
            
            # Re-apply freezing after loading if needed
            if freeze_backbone:
                for param in self.vision_model.parameters():
                    param.requires_grad = False

    def forward(self, image):
        """
        Forward pass of the perception model
        
        Args:
            image: Input image tensor
            
        Returns:
            List of perception variable logits (traffic safety, social safety, beautiful)
        """
        # Extract features from image
        with torch.set_grad_enabled(not self.freeze_backbone):
            if self.model_type == "convnextv2_tiny":
                # Get logits from the classification model (1000-dim)
                outputs = self.vision_model(image)
                feature_map = outputs.logits  # [batch_size, 1000]
            elif self.model_type == "repvit_m2_3":
                # Forward pass through the model, this will use our Identity head
                feature_map = self.vision_model(image)  # [batch_size, 640]
            elif self.model_type == "efficientvit_m1":
                # Process with EfficientViT
                feature_map = self.vision_model(image)  # [batch_size, 192]
            else:
                # Original DeiT model
                feature_map = self.vision_model(image)
        
        # Apply projection to desired dimension
        feature_map = self.feature_projector(feature_map)
        feature_map = self.feature_bn(feature_map)
        feature_map = self.feature_relu(feature_map)
        
        # If we have a single image, add a dimension
        if len(feature_map.shape) == 1:
            feature_map = feature_map.unsqueeze(0)
        
        # Apply input dropout for regularization
        feature_map = self.input_dropout(feature_map)
        
        # Apply stochastic depth during training (randomly skip feature extractors)
        if self.training and self.use_stochastic_depth:
            # Extract features for each perception variable with stochastic depth
            features = []
            for i, extractor in enumerate(self.feature_extractors):
                # Randomly drop the entire feature extractor with probability = stochastic_depth
                if torch.rand(1).item() > self.stochastic_depth:
                    features.append(extractor(feature_map))
                else:
                    # Skip this feature extractor
                    features.append(feature_map)  # Just use the input features
        else:
            # Normal forward pass during inference
            features = [extractor(feature_map) for extractor in self.feature_extractors]
        
        # Apply final classification layers
        perceptions = [classifier(features[i]) for i, classifier in enumerate(self.classifiers)]
        
        return perceptions

    def predict(self, image):
        """
        Predict perception scores for an image
        
        Args:
            image: Input image tensor
            
        Returns:
            List of perception scores
        """
        self.eval()
        with torch.no_grad():
            # Get logits
            logits = self.forward(image)
            
            # For each perception class, get the predicted level (argmax)
            predictions = [torch.argmax(logit, dim=1) + 1 for logit in logits]  # +1 to get 1-5 scale
            
        return predictions 


class SinglePerceptionModel(nn.Module):
    """
    Model for predicting a single perception variable from images
    Simpler architecture that focuses on one perception task
    """

    def __init__(self, perception_type, ordinal_levels=5, hidden_layers=2, 
                 hidden_dims=[512, 256], dropout_rates=[0.3, 0.2], 
                 freeze_backbone=False, path_pretrained_model=None,
                 model_type="deit_base"):
        """
        Initialize a single perception model
        
        Args:
            perception_type (str): Type of perception to predict ('traffic_safety', 'social_safety', or 'beautiful')
            ordinal_levels (int): Number of ordinal levels for the perception variable (default: 5)
            hidden_layers (int): Number of hidden layers
            hidden_dims (list): Dimensions of hidden layers
            dropout_rates (list): Dropout rates for each hidden layer
            freeze_backbone (bool): Whether to freeze the vision model backbone
            path_pretrained_model (str): Path to a pretrained model to load weights from
            model_type (str): Type of model to use as backbone ("deit_base", "convnextv2_tiny", "repvit_m2_3", or "efficientvit_m1")
        """
        super(SinglePerceptionModel, self).__init__()
        
        # Store perception type
        self.perception_type = perception_type
        
        # Define the CNN backbone based on model_type
        self.model_type = model_type
        
        if model_type == "convnextv2_tiny":
            # ConvNextV2 model that returns features instead of classification logits
            self.vision_model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-22k-224")
            self.feature_dim = 1000  # Output dimension matches DeiT model
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Add a feature projection layer
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        elif model_type == "repvit_m2_3":
            # RepVit model from timm
            base_model = timm.create_model('repvit_m2_3.dist_450e_in1k', pretrained=True)
            
            # Extract the feature dimension from the classifier
            # For RepVit models, the classifier is a Classfier with an embedding_size in the source code
            # Looking at the source code, we see it's the last layer's output dimension, which is 640 for m2_3
            self.feature_dim = 640  # Fixed dimension for repvit_m2_3 based on source code
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Create a new model without the classifier head
            self.vision_model = timm.create_model(
                'repvit_m2_3.dist_450e_in1k', 
                pretrained=True,
                num_classes=0  # This removes the classifier head
            )
            
            # Add a feature projection layer
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        elif model_type == "efficientvit_m1":
            # EfficientViT model from timm
            # Create a model that returns features only (no classification head)
            self.vision_model = timm.create_model(
                'efficientvit_m1.r224_in1k',
                pretrained=True,
                num_classes=0  # Remove classifier head
            )
            
            # For EfficientViT M1, the feature dimension is 192
            self.feature_dim = 192
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # Add a feature projection layer
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        else:
            # Default DeiT model
            self.vision_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=True)
            self.feature_dim = 1000  # Output dimension of the model
            self.projected_dim = hidden_dims[0] if hidden_dims and len(hidden_dims) > 0 else 512
            
            # For consistency, also add projection for DeiT
            self.feature_projector = nn.Linear(self.feature_dim, self.projected_dim)
            self.feature_bn = nn.BatchNorm1d(self.projected_dim)
            self.feature_relu = nn.ReLU()
        
        # Freeze the vision model if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Ensure hidden_dims and dropout_rates have the correct length
        if len(hidden_dims) < hidden_layers:
            hidden_dims = hidden_dims + [hidden_dims[-1]] * (hidden_layers - len(hidden_dims))
        if len(dropout_rates) < hidden_layers:
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (hidden_layers - len(dropout_rates))
            
        # Truncate if longer than needed
        hidden_dims = hidden_dims[:hidden_layers]
        dropout_rates = dropout_rates[:hidden_layers]
        
        # Store architecture parameters
        self.hidden_layers = hidden_layers
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        
        # Define the feature extraction head
        layers = []
        input_dim = self.projected_dim  # Use projected dimension
        
        # Build the entire network as a single sequential model
        for i in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))
            input_dim = hidden_dims[i]
        
        # Final classification layer
        layers.append(nn.Linear(hidden_dims[-1], ordinal_levels))
        
        # Create sequential model
        self.perception_head = nn.Sequential(*layers)
        
        # Load pretrained model if provided
        if path_pretrained_model is not None:
            self.load_state_dict(torch.load(path_pretrained_model), strict=True)
            
            # Re-apply freezing after loading if needed
            if freeze_backbone:
                for param in self.vision_model.parameters():
                    param.requires_grad = False

    def forward(self, image):
        """
        Forward pass of the perception model
        
        Args:
            image: Input image tensor
            
        Returns:
            Perception variable logits
        """
        # Extract features from image
        with torch.set_grad_enabled(not self.freeze_backbone):
            if self.model_type == "convnextv2_tiny":
                # Get logits from the classification model (1000-dim)
                outputs = self.vision_model(image)
                feature_map = outputs.logits  # [batch_size, 1000]
            elif self.model_type == "repvit_m2_3":
                # Forward pass through the model, this will use our Identity head
                feature_map = self.vision_model(image)  # [batch_size, 640]
            elif self.model_type == "efficientvit_m1":
                # Process with EfficientViT
                feature_map = self.vision_model(image)  # [batch_size, 192]
            else:
                # Original DeiT model
                feature_map = self.vision_model(image)
        
        # Apply projection to desired dimension
        feature_map = self.feature_projector(feature_map)
        feature_map = self.feature_bn(feature_map)
        feature_map = self.feature_relu(feature_map)
        
        feature_map = torch.squeeze(feature_map)
        
        # If we have a single image, add a dimension
        if len(feature_map.shape) == 1:
            feature_map = feature_map.unsqueeze(0)
            
        # Apply perception head
        perception = self.perception_head(feature_map)
        
        return perception

    def predict(self, image):
        """
        Predict perception scores for an image
        
        Args:
            image: Input image tensor
            
        Returns:
            Perception scores
        """
        self.eval()
        with torch.no_grad():
            # Get logits
            logits = self.forward(image)
            
            # Get the predicted level (argmax)
            predictions = torch.argmax(logits, dim=1) + 1  # +1 to get 1-5 scale
            
        return predictions 