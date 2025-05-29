import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
# 1. Veri oluştur
data = {
    "vize": [],
    "final": [],
    "gecti_mi": []
}

for _ in range(200):  # 200 satır veri için
    vize_notu = random.randint(0, 100)
    final_notu = random.randint(0, 100)
    data["vize"].append(vize_notu)
    data["final"].append(final_notu)

    if vize_notu*0.4 + final_notu*0.6 >= 50:
        data["gecti_mi"].append(1)
    else:
        data["gecti_mi"].append(0)




df = pd.DataFrame(data)

# 2. X (giriş) ve y (çıktı) ayrımı
X = df[["vize", "final"]]  # Girişler
y = df["gecti_mi"]         # Etiket

# 3. Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 4. Modeli tanımla ve eğit
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Test verisi ile tahmin yap
tahminler = model.predict(X_test)

# 6. Sonuçları yazdır
print("Gerçek:", y_test.tolist())
print("Tahmin:", tahminler.tolist())

# 7. Yeni veriyle tahmin (örnek)
yeni_ogrenci = pd.DataFrame([[49, 50]], columns=["vize", "final"])
sonuc = model.predict(yeni_ogrenci)
print("Yeni öğrenci geçti mi?", "Evet" if sonuc[0] == 1 else "Hayır")

