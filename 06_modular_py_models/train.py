
"""
Trains a model given path for data

example:
python train.py DATA_PATH
"""

import os
import sys
import pathlib
import torch
from torch import nn
from torchvision import transforms
import data_download, data_setup, model_builder, engine

def main():
    if len(sys.argv) < 2:
        print("Please provide a path")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = sys.argv[1]

    train_dir, eval_dir = data_download.download_sample_to_path(data_path)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_loader, eval_loader, class_names = data_setup.create_dataloaders(train_dir, eval_dir,
                                                                           transform, 32, 0)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    model_0 = model_builder.TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Train model_0
    model_0_results = engine.train(model=model_0,
                            train_dataloader=train_loader,
                            test_dataloader=eval_loader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS,
                            device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model
    model_save_path = pathlib.Path(data_path).parent / "models"
    if not model_save_path.is_dir():
        model_save_path.mkdir()

    engine.save_model(model = model_0, target_dir = model_save_path,
                      model_name = "6_modular_py_models_cells_tinyvgg.pth")


if __name__ == "__main__":
    main()
