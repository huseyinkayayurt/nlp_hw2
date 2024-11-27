from data_loader import load_data_from_csv
import random
import re
from llm_evaluation import evaluate_llm


def main():
    trsav1_file_path = "data_set/TRSAv1.csv"
    ttc4900_file_path = "data_set/ttc4900.csv"

    data_set_trsav1 = load_data_from_csv(trsav1_file_path, "review", "score")
    data_set_ttc4900 = load_data_from_csv(ttc4900_file_path, "text", "category")

    indices = random.sample(range(len(data_set_trsav1)), 50)
    selected_trsav1 = [data_set_trsav1[i] for i in indices]
    for item in selected_trsav1:
        item['review'] = re.sub(r'[^\w\s.,!?]', '', item['review'])

    indices = random.sample(range(len(data_set_ttc4900)), 50)
    selected_ttc4900 = [data_set_ttc4900[i] for i in indices]
    for item in selected_ttc4900:
        item['text'] = re.sub(r'[^\w\s.,!?]', '', item['text'])
        item['text'] = item['text'].strip()

    train_dataset_trsav1 = selected_trsav1[:40]
    test_dataset_trsav1 = selected_trsav1[40:]

    train_dataset_ttc4900 = selected_ttc4900[:40]
    test_dataset_ttc4900 = selected_ttc4900[40:]

    llm_models = [
        "local_models/kanarya-750m",
        "local_models/llama3-8b-tr",
        "local_models/OpenHermes-2.5-Mistral-7B",
    ]

    for model in llm_models:
        results = evaluate_llm(
            model_name=model,
            train_dataset=train_dataset_ttc4900,
            test_dataset=test_dataset_ttc4900,
            text_key="text",
            category_key="category",
            shots_list=[0, 7, 14]
        )
        print("Sonuçlar:", results)

        results = evaluate_llm(
            model_name=model,
            train_dataset=train_dataset_trsav1,
            test_dataset=test_dataset_trsav1,
            text_key="review",
            category_key="score",
            shots_list=[0, 3, 6]
        )
        print("Sonuçlar:", results)


if __name__ == "__main__":
    main()
