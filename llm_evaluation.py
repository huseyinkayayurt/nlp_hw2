from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # İlerleme çubuğu için
import random
import os

# TOKENIZERS_PARALLELISM uyarısını devre dışı bırak
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_few_shot_prompt(train_dataset, text, shots, max_prompt_length=1500):
    """
    Few-shot prompt oluşturur.
    """
    # Görev açıklamasını ekle
    task_description = "Aşağıdaki metinlerin kategori etiketlerini belirleyin:\n\n"

    # Few-shot örneklerini rastgele seç
    train_dataset_list = list(train_dataset)
    few_shot_examples = random.sample(train_dataset_list, shots)

    # Few-shot örneklerini formatla
    examples = ""
    for example in few_shot_examples:
        example_text = f"Metin: {example['text']}\nKategori: {example['category']}\n\n"
        if len(task_description) + len(examples) + len(example_text) <= max_prompt_length:
            examples += example_text
        else:
            break

    # Test verisi için son kısmı ekle
    test_prompt = f"Test metni:\nMetin: {text}\nKategori:"

    # Tüm prompt'u birleştir
    prompt = task_description + examples + test_prompt
    return prompt


def evaluate_llm(model_name, train_dataset, test_dataset, shots=None):
    """
    LLM modelini zero-shot ve few-shot öğrenme için değerlendirir.

    Args:
        model_name (str): Hugging Face model adı.
        train_dataset (Dataset): Eğitim veri seti.
        test_dataset (Dataset): Test veri seti.
        shots (list): Few-shot örnek sayıları.

    Returns:
        dict: Shot bazlı doğruluk oranları.
    """
    if shots is None:
        shots = [0]
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Pad token ayarı
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model yüklemesi tamamlandı: {model_name}\n")

    results = {}
    for shot in shots:
        print(f"{shot}-shot değerlendirmesi başlıyor...")
        correct = 0

        # Test veri seti üzerinde tahminler için tqdm ile ilerleme çubuğu
        for test_example in tqdm(test_dataset, desc=f"{shot}-shot değerlendirmesi"):
            text = test_example["text"]
            true_category = test_example["category"]

            # Prompt oluşturma
            if shot > 0:
                prompt = prepare_few_shot_prompt(train_dataset, text, shot)
            else:
                prompt = f"Soru: {text}\nCevap:"

            # Tokenizer ile giriş verisi oluştur
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            # Tahmin yap
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=256,  # Maksimum üretilecek token sayısı
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Tahminin doğru olup olmadığını kontrol et
            if str(true_category) in predicted:
                correct += 1

        # Doğruluk oranını hesapla
        accuracy = correct / len(test_dataset)
        results[f"{shot}-shot"] = accuracy

        print(f"{shot}-shot doğruluk oranı: {accuracy:.2f}\n")

    return results
