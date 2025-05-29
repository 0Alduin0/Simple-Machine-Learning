import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random


#Tüm bir iterasyon yaptıktan sonra iterasyonları karşılaktırmak için kullanılıyor
sonuçlar=[]


#11 kere 2000 lik veri setinden sonuç çıkaracak
while(True):
    #Veriler oluşturuluyor
    data = {
        "vize": [],
        "final": [],
        "gecti_mi": []
    }

    for _ in range(2000):  #2000 adet veri verilecek
        vize_notu = random.randint(0, 100)
        final_notu = random.randint(0, 100)
        data["vize"].append(vize_notu)
        data["final"].append(final_notu)

        if vize_notu*0.4 + final_notu*0.6 >= 50:
            data["gecti_mi"].append(1)
        else:
            data["gecti_mi"].append(0)



    #veri tabloya dönüştürülüyor
    df = pd.DataFrame(data)

    # X (sınav notları) ve y (sınav notlarına göre geçme kalma durumu)
    X = df[["vize", "final"]]  # Girişler
    y = df["gecti_mi"]         # Etiket


    # Verileri eğitim ve test olacak şekilde bölüyoruz.
    # Train ile bitenlerle öğrenme test ile bitenlerle alıştırma yapıyor.
    # Test_size ise verilen 2000 adet verinin yüzde kaçıyla test yapcağımızı söylüyor.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # LogisticRegression sınıflandımra için kullanılan bir algoritmadır.
    # Bu proje gibi geçme kalma durumlarında iyi çalışır.
    model = LogisticRegression()

    # modele veriler giriliyor
    model.fit(X_train, y_train)

    # Test verisi ile tahmin yapılıyor
    tahminler = model.predict(X_test)

    # Sonuçları yazdırıyoruz 0 kalma 1 geçme demek kullanmak için baştaki işareti silin
    #print("Gerçek:", y_test.tolist())
    #print("Tahmin:", tahminler.tolist())

    # kendi verdiğmiz veriyle algoritmayı test ediyoruz
    girilen_vize=input("Vize Notu: ")
    girilen_final=input("Final Notu: ")
    yeni_ogrenci = pd.DataFrame([[girilen_vize, girilen_final]], columns=["vize", "final"])
    sonuc = model.predict(yeni_ogrenci)
    print("Öğrenci geçti mi?", "Evet\n" if sonuc[0] == 1 else "Hayır\n")
    sonuçlar.append(sonuc)



