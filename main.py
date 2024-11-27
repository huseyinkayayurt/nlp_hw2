from data_loader import load_and_prepare_data
from llm_evaluation import evaluate_llm
from visualizer import plot_results


def main():
    # Veri setlerini yükle ve hazırla
    trsav1_train, trsav1_test = load_and_prepare_data(
        file_path="data_set/TRSAv1.csv",
        text_key="review",
        category_key="score",
        sample_size=50
    )
    ttc4900_train, ttc4900_test = load_and_prepare_data(
        file_path="data_set/ttc4900.csv",
        text_key="text",
        category_key="category",
        sample_size=50
    )

    # Değerlendirilecek modeller
    llm_models = [
        "local_models/kanarya-750m",
        "local_models/llama3-8b-tr",
        "local_models/OpenHermes-2.5-Mistral-7B",
    ]
    # Modelleri değerlendir
    for model in llm_models:
        print(f"Model değerlendiriliyor: {model}")

        # TTC4900 değerlendirmesi
        ttc4900_results = evaluate_llm(
            model_name=model,
            train_dataset=ttc4900_train,
            test_dataset=ttc4900_test,
            text_key="text",
            category_key="category",
            shots_list=[0, 7, 14]
        )
        print("TTC4900 Sonuçlar:", ttc4900_results)
        plot_results(ttc4900_results, f"ttc4900_{model.split('/')[-1]}")

        # TRSAv1 değerlendirmesi
        trsav1_results = evaluate_llm(
            model_name=model,
            train_dataset=trsav1_train,
            test_dataset=trsav1_test,
            text_key="review",
            category_key="score",
            shots_list=[0, 3, 6]
        )
        print("TRSAv1 Sonuçlar:", trsav1_results)
        plot_results(trsav1_results, f"trsav1_{model.split('/')[-1]}")


if __name__ == "__main__":
    main()
