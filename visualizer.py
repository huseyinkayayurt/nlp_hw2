import matplotlib.pyplot as plt
import os


def plot_results_for_dataset(dataset_name, results_dict):
    """
    Bir veri seti için tüm modellerin sonuçlarını tek bir grafikte gösterir.

    Args:
        dataset_name (str): Veri setinin adı.
        results_dict (dict): Her modelin shot doğruluk oranlarını içeren sözlük.
                             Örn: {"model1": {"0-shot": 0.5, "7-shot": 0.6}, "model2": {...}}
    """
    plt.figure(figsize=(10, 6))

    for model_name, results in results_dict.items():
        shots = list(results.keys())
        accuracies = list(results.values())
        plt.plot(shots, accuracies, marker="o", linestyle="-", label=model_name)

    # Grafik özellikleri
    plt.title(f"Shot Accuracy Comparison for {dataset_name}")
    plt.xlabel("Shots")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Grafik dosyasını kaydetme
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{dataset_name}_accuracy_comparison.png")
    plt.close()
