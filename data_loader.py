from datasets import load_dataset
import random


def load_and_split_dataset(dataset_name, sample_size=None, train_ratio=0.8):
    """
    Veri setini yükler, opsiyonel olarak örneklem alır ve eğitim/test olarak böler.

    Args:
        dataset_name (str): Hugging Face veri seti adı.
        sample_size (int or None): Rastgele alınacak veri sayısı (None ise tüm veriyi kullanır).
        train_ratio (float): Eğitim veri oranı (0-1 arasında bir değer).

    Returns:
        tuple: Eğitim ve test veri setleri (train_dataset, test_dataset).
    """
    dataset = load_dataset(dataset_name, split="train")  # Varsayılan split 'train'

    # Eğer örnekleme yapılacaksa
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Eğitim ve test veri setini ayır
    train_size = int(len(dataset) * train_ratio)
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, test_dataset
