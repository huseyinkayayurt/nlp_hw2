from data_loader import load_and_split_dataset
from llm_evaluation import evaluate_llm
import time


def main():
    print("Veri setleri yükleniyor ve ayrılıyor...\n")
    start_time = time.time()  # Başlangıç zamanı

    # Veri setlerini yükle ve eğitim/test olarak ayır
    print("1. Veri seti: savasy/ttc4900")
    ttc4900_train, ttc4900_test = load_and_split_dataset("savasy/ttc4900", sample_size=49)
    print(f"TTC4900 - Eğitim Seti Boyutu: {len(ttc4900_train)}, Test Seti Boyutu: {len(ttc4900_test)}\n")

    # ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1
    # burak/Trendyol-Turkcell-7b-mixture
    # ytu-ce-cosmos/turkish-gpt2-large
    # ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1
    # asafaya/kanarya-750m
    # berkecr/tr-dare-merge-7B
    # ---

    # LLM modelini kullanarak performans ölçümü
    model_name = "asafaya/kanarya-750m"
    print(f"Model değerlendirmesi başlatılıyor: {model_name}")
    results = evaluate_llm(model_name, ttc4900_train, ttc4900_test, shots=[0, 7, 14])

    # Sonuçları yazdır
    for shot, accuracy in results.items():
        print(f"{shot}-shot Accuracy: {accuracy:.2f}")

    # print("2. Veri seti: maydogan/TRSAv1")
    # trsav1_train, trsav1_test = load_and_split_dataset("maydogan/TRSAv1", sample_size=5000)
    # print(f"TRSAv1 - Eğitim Seti Boyutu: {len(trsav1_train)}, Test Seti Boyutu: {len(trsav1_test)}\n")

    # Çalışma süresi hesapla
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Toplam çalışma süresi: {int(hours)} saat, {int(minutes)} dakika, {seconds:.2f} saniye")


if __name__ == "__main__":
    main()
