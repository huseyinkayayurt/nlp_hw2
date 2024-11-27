import matplotlib.pyplot as plt
import os


def plot_results(results, model_name):
    """
    Shot doğruluk oranlarını grafiğe çizer ve kaydeder.

    Args:
        results (dict): Shot doğruluk oranları.
        model_name (str): Modelin adı.
    """
    shots = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 6))
    plt.plot(shots, accuracies, marker="o", linestyle="-", label="Accuracy")
    plt.title(f"Shot Accuracy for {model_name}")
    plt.xlabel("Shots")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_name}_accuracy.png")
    plt.close()
