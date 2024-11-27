from datasets import load_dataset
import csv


def load_and_split_dataset(dataset_name, sample_size=None, train_ratio=0.8):
    dataset = load_dataset(dataset_name, split="train")  # Varsayılan split 'train'

    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    train_size = int(len(dataset) * train_ratio)
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, len(dataset)))

    return train_dataset, test_dataset


def load_data_from_csv(file_path, text_header, category_header):
    text = []
    category = []
    data_set = []

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_set.append({text_header: row[text_header], category_header: row[category_header]})
            text.append(row[text_header])
            category.append(row[category_header])

    return data_set
