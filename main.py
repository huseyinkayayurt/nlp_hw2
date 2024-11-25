from data_loader import load_and_split_dataset
from llm_evaluation import evaluate_llm,load_llm_model,evaluate_llm_optimized
from visulation import save_graph
import time


def main():
    print("Veri setleri yükleniyor ve ayrılıyor...\n")

    # Veri setlerini yükle ve ayır
    ttc4900_train, ttc4900_test = load_and_split_dataset("savasy/ttc4900", sample_size=None)
    trsav1_train, trsav1_test = load_and_split_dataset("maydogan/TRSAv1", sample_size=5000)

    print(f"TTC4900 - Eğitim Seti Boyutu: {len(ttc4900_train)}, Test Seti Boyutu: {len(ttc4900_test)}\n")
    print(f"TRSAv1 - Eğitim Seti Boyutu: {len(trsav1_train)}, Test Seti Boyutu: {len(trsav1_test)}\n")

    # LLM modelini kullanarak performans ölçümü
    model_name = "./local_models/kanarya-750m"
    model_name = "asafaya/kanarya-750m"
    tokenizer,model=load_llm_model(model_name)
    results_kanarya_ttc4900 = evaluate_llm_optimized(model,tokenizer, ttc4900_train, ttc4900_test,"text","category", shots=[0, 7, 14])
    save_graph(results_kanarya_ttc4900, "ttc4900_kanarya-750m")
    # Sonuçları yazdır
    for shot, accuracy in results_kanarya_ttc4900.items():
        print(f"{shot}-shot Accuracy: {accuracy:.2f}")


    # results_kanarya_trsav1 = evaluate_llm(model,tokenizer, trsav1_train, trsav1_test,"review","score", shots=[0, 3, 6])
    # save_graph(results_kanarya_trsav1, "trsav1_kanarya-750m")
    # # Sonuçları yazdır
    # for shot, accuracy in results_kanarya_trsav1.items():
    #     print(f"{shot}-shot Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    start_time = time.time()  # Başlangıç zamanı
    main()
    # Çalışma süresi hesapla
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Toplam çalışma süresi: {int(hours)} saat, {int(minutes)} dakika, {seconds:.2f} saniye")
