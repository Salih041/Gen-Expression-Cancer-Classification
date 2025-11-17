import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    #! VERİ HAZIRLAMA
    # Veri dosyalarını yükleme
    data_df = pd.read_csv("data.csv")
    labels_df = pd.read_csv("labels.csv")
    print(f"Veri: data.csv {data_df.shape}")
    print(f"Etiketler: labels.csv {labels_df.shape}")

    # Veri ve etiketleri birlşetirme
    df_full = pd.merge(data_df, labels_df, on='Unnamed: 0')
    print(f"\nYeni Boyut: {df_full.shape}")

    # Kopya kontrölü
    kopya_sayisi = df_full.duplicated().sum()
    print(f"Kopya satır sayısı: {kopya_sayisi}")
    if kopya_sayisi > 0:
        print("Kopyalar atılıyor...")
        df_full = df_full.drop_duplicates(ignore_index=True)
        print(f"Temiz boyut: {df_full.shape}")

    # Öznitelik ve target ayırma
    y = df_full['Class']
    x = df_full.drop(columns=['Unnamed: 0', 'Class'])
    
    print(f"\nx (öznitelik): {x.shape}")
    print(f"y (hedef): {y.shape}")

    # Hedef (y) kodlama
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)


    #! BÖLME VE ÖLÇEKLENDİRME
    print("\n--- AŞAMA 2 BAŞLATILIYOR: Bölme ve Ölçeklendirme ---")

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.20, stratify=y,random_state=42)
    # 20-80

    # Ölçeklendirme
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    #! MODEL 1 LOGISTIK REGRESSION
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    print("Lojistik Regresyon eğitiliyor")
    lr_model.fit(x_train_scaled, y_train)
    print("Model eğitildi.")

    y_pred_lr = lr_model.predict(x_test_scaled)
    print("Test yapıldı")

    print("\n!!!! LOJİSTİK REGRESYON SONUÇLARI !!!!")
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Genel Doğruluk (Accuracy): {accuracy_lr * 100:.2f}%")

    print("\nSınıflandırma Raporu:")
    report_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_)#le.classes : (kod->isim)
    print(report_lr)

except FileNotFoundError:
    print("HATA: Dosyalar bulunamadı")
except Exception as e:
    print(f"HATA: {e}")



