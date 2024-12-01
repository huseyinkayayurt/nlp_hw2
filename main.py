from data_loader import load_and_prepare_data
from llm_evaluation import evaluate_llm
from visualizer import plot_results_for_dataset


def main():
    # Veri setlerini yükle ve hazırla
    trsav1_train, trsav1_test = load_and_prepare_data(
        file_path="data_set/TRSAv1.csv",
        text_key="review",
        category_key="score",
        sample_size=5000
    )
    ttc4900_train, ttc4900_test = load_and_prepare_data(
        file_path="data_set/ttc4900.csv",
        text_key="text",
        category_key="category",

    )

    # Değerlendirilecek modeller
    # llm_models = [
    #     "local_models/Turkish-Llama-8b-DPO-v0.1",
    #     "local_models/VeriUS-LLM-8b-v0.2",
    #     "local_models/llama3-8b-tr",
    # ]
    llm_models = [
        "local_models/new/kanarya-750m",
        "local_models/new/Phi-3-mini-4k-instruct",
        "local_models/new/turkish-gpt2-lage",
    ]

    # Veri seti ve model sonuçları için sözlükler
    trsav1_results = {}
    ttc4900_results = {}

    for model in llm_models:
        model_name = model.split("/")[-1]

        # TTC4900 değerlendirmesi
        print(f"TTC4900 veri seti değerlendiriliyor: {model_name}")
        ttc4900_results[model_name] = evaluate_llm(
            model_name=model,
            train_dataset=ttc4900_train,
            test_dataset=ttc4900_test,
            text_key="text",
            category_key="category",
            shots_list=[0, 7, 14]
        )

        # TRSAv1 değerlendirmesi
        print(f"TRSAv1 veri seti değerlendiriliyor: {model_name}")
        trsav1_results[model_name] = evaluate_llm(
            model_name=model,
            train_dataset=trsav1_train,
            test_dataset=trsav1_test,
            text_key="review",
            category_key="score",
            shots_list=[0, 3, 6]
        )

    print(ttc4900_results)
    print(trsav1_results)
    # Grafik oluşturma
    plot_results_for_dataset("TTC4900", ttc4900_results)
    plot_results_for_dataset("TRSAv1", trsav1_results)


if __name__ == "__main__":
    main()
