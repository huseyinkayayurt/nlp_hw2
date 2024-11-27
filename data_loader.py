import pandas as pd
import random
import re


def load_data_from_csv(file_path, text_key, category_key):
    """
    CSV dosyasını yükler ve dictionary formatında döndürür.

    Args:
        file_path (str): CSV dosyasının yolu.
        text_key (str): Metin içeriğinin olduğu sütun adı.
        category_key (str): Kategori bilgilerinin olduğu sütun adı.

    Returns:
        list[dict]: Veri seti elemanları.
    """
    data = pd.read_csv(file_path)
    data = data[[text_key, category_key]].dropna()
    return data.to_dict(orient="records")


def clean_text_fields(dataset, text_key):
    """
    Metin alanlarını temizler: özel karakterleri kaldırır ve boşlukları kırpar.

    Args:
        dataset (list[dict]): Veri seti.
        text_key (str): Temizlenecek metin alanının anahtarı.

    Returns:
        list[dict]: Temizlenmiş veri seti.
    """
    for item in dataset:
        item[text_key] = re.sub(r"[^\w\s.,!?]", "", item[text_key]).strip()
    return dataset


def load_and_prepare_data(file_path, text_key, category_key, sample_size):
    """
    Veriyi yükler, temizler ve eğitim/test için böler.

    Args:
        file_path (str): Veri dosyasının yolu.
        text_key (str): Metin içeriği sütunu.
        category_key (str): Kategori sütunu.
        sample_size (int): Örnek veri boyutu.

    Returns:
        tuple: Eğitim ve test veri kümeleri.
    """
    dataset = load_data_from_csv(file_path, text_key, category_key)
    dataset = clean_text_fields(dataset, text_key)
    sampled_dataset = random.sample(dataset, sample_size)
    train_data = sampled_dataset[:int(0.8 * sample_size)]
    test_data = sampled_dataset[int(0.8 * sample_size):]
    return train_data, test_data
