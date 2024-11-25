from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm  # İlerleme çubuğu için
import random
import os
import torch

# TOKENIZERS_PARALLELISM uyarısını devre dışı bırak
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_few_shot_prompt(train_dataset, text, shots,text_header,category_header, max_prompt_length=1500):
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
        example_text = f"Metin: {example[text_header]}\nKategori: {example[category_header]}\n\n"
        if len(task_description) + len(examples) + len(example_text) <= max_prompt_length:
            examples += example_text
        else:
            break

    # Test verisi için son kısmı ekle
    test_prompt = f"Test metni:\nMetin: {text}\nKategori:"

    # Tüm prompt'u birleştir
    prompt = task_description + examples + test_prompt
    return prompt

def load_llm_model(model_name):
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Padding ayarları
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Sol-padding olarak ayarlandı

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print(f"Model yüklemesi tamamlandı: {model_name}\n")

    return tokenizer, model


def evaluate_llm(model,tokenizer,train_dataset, test_dataset,text_header,category_header, shots=None,):
    """
    LLM modelini zero-shot ve few-shot öğrenme için değerlendirir.

    Args:
        train_dataset (Dataset): Eğitim veri seti.
        test_dataset (Dataset): Test veri seti.
        shots (list): Few-shot örnek sayıları.

    Returns:
        dict: Shot bazlı doğruluk oranları.
    """
    if shots is None:
        shots = [0]


    results = {}
    for shot in shots:
        print(f"{shot}-shot değerlendirmesi başlıyor...")
        correct = 0

        # Test veri seti üzerinde tahminler için tqdm ile ilerleme çubuğu
        for test_example in tqdm(test_dataset, desc=f"{shot}-shot değerlendirmesi"):
            text = test_example[text_header]
            true_category = test_example[category_header]

            # Prompt oluşturma
            if shot > 0:
                prompt = prepare_few_shot_prompt(train_dataset, text,shot,text_header,category_header)
            else:
                prompt = f"Soru: {text}\nCevap:"

            # Tokenizer ile giriş verisi oluştur
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(device)
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

def evaluate_llm_optimized(
        model,
        tokenizer,
        train_dataset,
        test_dataset,
        text_header,
        category_header,
        shots=None,
        batch_size=32):
    if shots is None:
        shots = [0]

    results = {}
    for shot in shots:
        print(f"{shot}-shot değerlendirmesi başlıyor...")
        correct = 0
        batch_prompts = []

        # Test veri setini batch'lere böl

        for idx, test_example in tqdm(enumerate(test_dataset), desc=f"{shot}-shot değerlendirmesi"):
            text = test_example[text_header]
            true_category = test_example[category_header]

            if shot > 0:
                prompt = prepare_few_shot_prompt(train_dataset, text, shot, text_header, category_header)
            else:
                prompt = f"Soru: {text}\nCevap:"

            batch_prompts.append(prompt)

            # Eğer batch dolduysa işle
            if len(batch_prompts) == batch_size or idx == len(test_dataset) - 1:
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=64,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

                # Doğruluğu kontrol et
                for i, predicted in enumerate(decoded_outputs):
                    if str(test_dataset[i][category_header]) in predicted:
                        correct += 1

                batch_prompts = []

        accuracy = correct / len(test_dataset)
        results[f"{shot}-shot"] = accuracy
        print(f"{shot}-shot doğruluk oranı: {accuracy:.2f}\n")

    return results
