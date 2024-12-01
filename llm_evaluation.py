import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch


def prepare_few_shot_prompt(dataset, text, text_key, category_key, shots=0, max_prompt_length=1500):
    """
    Few-shot prompt hazırlama.
    """
    task_description = "Aşağıdaki metinlerin kategori etiketlerini belirleyin:\n\n"
    examples = ""
    if shots > 0:
        sampled_examples = random.sample(list(dataset), shots)
        for example in sampled_examples:
            example_text = f"Metin: {example[text_key][:500]}\nKategori: {example[category_key]}\n\n"
            if len(task_description) + len(examples) + len(example_text) <= max_prompt_length:
                examples += example_text
            else:
                break
    test_prompt = f"Test metni:\nMetin: {text[:500]}\nKategori:"
    return task_description + examples + test_prompt


def evaluate_llm(model_name, train_dataset, test_dataset, text_key, category_key, shots_list=None):
    """
    LLM modelini değerlendirir.
    """
    if shots_list is None:
        shots_list = [0]  # Varsayılan: Zero-shot

    # Model ve tokenizer yükleme
    print(f"Model yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()

    results = {}
    for shots in shots_list:
        print(f"{shots}-shot değerlendirmesi başlıyor...")
        correct = 0
        for test_example in tqdm(test_dataset, desc=f"{shots}-shot değerlendirmesi"):
            text = test_example[text_key]
            true_category = test_example[category_key]
            prompt = prepare_few_shot_prompt(train_dataset, text, text_key, category_key, shots)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=2048)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if str(true_category).lower() in predicted.lower():
                correct += 1
        accuracy = correct / len(test_dataset)
        results[f"{shots}-shot"] = accuracy
        print(f"{shots}-shot doğruluk oranı: {accuracy:.2f}\n")
    return results
