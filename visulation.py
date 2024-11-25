import os

import matplotlib.pyplot as plt

def save_graph(results, dataset_name, output_dir="graphs"):
    """
    Sonuçları grafikleştir ve dosyaya kaydet.
    :param results: Shot değerlendirme sonuçları {shot: accuracy} formatında.
    :param dataset_name: Veri seti ismi (ör. "ttc4900", "trsav1").
    :param output_dir: Grafiklerin kaydedileceği dizin.
    """
    # Grafik için renkler ve stil
    colors = ['blue', 'orange', 'green']
    styles = ['o-', 's-', '^-']

    plt.figure(figsize=(10, 6))

    # Her shot için grafiği oluştur
    for idx, (shot, accuracy) in enumerate(results.items()):
        plt.plot(shot, accuracy, styles[idx], label=f"{shot}-Shot", color=colors[idx])

    # Grafik düzenlemeleri
    plt.title(f"Shot Accuracy for {dataset_name}")
    plt.xlabel("Shot Count")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)  # Accuracy değerleri için 0-1 arası sınır
    plt.legend()
    plt.grid(True)

    # Grafik dosyasını kaydet
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{dataset_name}_shot_accuracy.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Grafik kaydedildi: {file_path}")