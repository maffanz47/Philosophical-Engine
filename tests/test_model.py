import torch
from src.model import PhiloClassifier

def test_philo_classifier_forward():
    input_dim = 200
    num_classes = 6
    batch_size = 4
    
    model = PhiloClassifier(input_dim=input_dim, num_classes=num_classes)
    
    # Create dummy input tensor
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    out = model(x)
    
    # Check output shape
    assert out.shape == (batch_size, num_classes)
    
    # Check that gradients can be computed
    loss = out.sum()
    loss.backward()
    
    # Ensure a parameter has a gradient
    assert list(model.parameters())[0].grad is not None
