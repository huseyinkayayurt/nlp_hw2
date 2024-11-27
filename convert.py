import pandas as pd


def convert_parquet_to_csv(parquet_file_path, csv_file_path):
    """
    Parquet dosyasını CSV formatına dönüştürüp kaydeder.

    Args:
        parquet_file_path (str): Parquet dosyasının yolu.
        csv_file_path (str): Çıktı olarak kaydedilecek CSV dosyasının yolu.
    """
    try:
        # Parquet dosyasını oku
        df = pd.read_parquet(parquet_file_path)

        # CSV dosyasını kaydet
        df.to_csv(csv_file_path, index=False)

        print(f"Başarıyla CSV'ye dönüştürüldü ve {csv_file_path} olarak kaydedildi.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
