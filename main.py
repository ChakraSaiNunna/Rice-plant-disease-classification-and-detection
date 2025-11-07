from src.data_loader import get_data
from src.model_builder import build_densenet_model
from src.train import train_model
from src.bat_optimizer import bat_algorithm

TRAIN_DIR = "C:\\Users\\narra\\Downloads\\MyProject\\archive\\RiceLeafsDisease\\train"
VAL_DIR = "C:\\Users\\narra\\Downloads\\MyProject\\archive\\RiceLeafsDisease\\validation"

if __name__ == "__main__":
    train_gen, val_gen = get_data(TRAIN_DIR, VAL_DIR)
    model = build_densenet_model()
    model, history = train_model(model, train_gen, val_gen)
    optimized_model = bat_algorithm(model, val_gen)
    loss, acc = optimized_model.evaluate(val_gen)
    print(f"✅ Final Validation Accuracy: {acc:.4f}")
