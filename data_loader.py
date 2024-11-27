import csv
import random


def load_and_prepare_data(file_path, text_key, category_key, sample_size=None):
    """
    Veri setini yükler, gerekli temizlemeleri yapar ve isteğe bağlı olarak örneklem alır.

    Args:
        file_path (str): Veri setinin dosya yolu.
        text_key (str): Metin alanının anahtarı.
        category_key (str): Kategori alanının anahtarı.
        sample_size (int, optional): Alınacak örneklem boyutu. None ise tüm veri seti alınır.

    Returns:
        tuple: Eğitim ve test veri setleri (train_dataset, test_dataset)
    """
    dataset = []

    # Veri setini yükle
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset.append({
                text_key: row[text_key].strip(),
                category_key: row[category_key].strip()
            })

    # Eğer örneklem boyutu belirtilmişse rastgele örnek seç
    if sample_size is not None and sample_size < len(dataset):
        dataset = random.sample(dataset, sample_size)

    # Veri setini %80 eğitim ve %20 test olarak ayır
    split_index = int(len(dataset) * 0.8)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]

    return train_dataset, test_dataset
