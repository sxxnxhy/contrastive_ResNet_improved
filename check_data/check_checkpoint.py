"""Check saved model"""
import torch

checkpoint = torch.load('models/ContrastiveModel.pth', map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

print("\nModel state dict keys:")
state_dict = checkpoint['model_state_dict']

# Check if classifier exists
has_classifier = any('classifier' in key for key in state_dict.keys())
print(f"\nHas classifier in saved model: {has_classifier}")

if has_classifier:
    print("\nClassifier weights:")
    for key in state_dict.keys():
        if 'classifier' in key:
            print(f"  {key}: {state_dict[key].shape}")
else:
    print("\n❌ No classifier in saved model!")
    print("   This is an old model trained without classification head")

print("\nAll model keys:")
for key in sorted(state_dict.keys()):
    print(f"  {key}")
