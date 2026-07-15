# Proyek Akhir: Prediksi Attrition Karyawan (Employee Retention)

Proyek *Data Science* untuk memprediksi risiko karyawan keluar (*attrition*) pada perusahaan "Jaya Jaya Maju" guna membantu manajemen SDM merancang strategi retensi yang berbasis data.

## Business Understanding

Jaya Jaya Maju adalah perusahaan multinasional dengan lebih dari 1000 karyawan. Perusahaan menghadapi tantangan *attrition rate* di atas 10%. Proyek ini bertujuan untuk:

1. Mengidentifikasi faktor-faktor yang berkontribusi terhadap *attrition*.
2. Membangun model klasifikasi untuk memprediksi kemungkinan karyawan keluar.
3. Memberikan rekomendasi strategi retensi karyawan.

## Dataset

- Sumber: [employee_data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/f4a7541bc3dfca0012e20778e135c03b3e76dc67/employee/employee_data.csv)
- Target: kolom `Attrition` (Ya/Tidak).
- *Attrition rate* pada dataset: **12.18%**.

## Struktur Direktori

```
.
├── submission-ds-proyek1.ipynb   # Notebook analisis & pemodelan
├── Laporan.md                    # Laporan lengkap (Business Understanding s.d. Rekomendasi)
├── attrition-model.pkl           # Model terlatih (Logistic Regression)
├── requirements.txt              # Dependencies
├── image.png, image-1.png, image-2.png  # Visualisasi dashboard
└── margaretalola-dashboard.txt   # Link dashboard Looker Studio
```

## Pemodelan

- **Algoritma:** Logistic Regression
- **Penanganan ketidakseimbangan:** SMOTE (dari `imbalanced-learn`)
- **Preprocessing:** `StandardScaler`, `LabelEncoder`, *winsorize* (scipy)
- **Evaluasi:** ROC AUC mencapai **0.784**, dilengkapi *classification report* dan *confusion matrix*

Faktor penting yang memengaruhi *attrition*: `Monthly Income`, `Work-Life Balance`, `Job Satisfaction`, `Overtime`, dan `Performance Rating`.

## Dashboard

Dashboard interaktif (Looker Studio): https://lookerstudio.google.com/reporting/1c303a6c-3232-4a0f-b9fc-13ee80d55bcd

## Teknologi

- `Python`, `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`
- `matplotlib`, `seaborn` (visualisasi)
- `joblib` (penyimpanan model)

## Cara Menjalankan

```bash
pip install -r requirements.txt
```

Buka dan jalankan `submission-ds-proyek1.ipynb` untuk mereplikasi analisis dan pelatihan model. Model hasil pelatihan tersimpan di `attrition-model.pkl`.

## Rekomendasi Action Items

- Pengembangan jalur karier & keterampilan (mentoring, suksesi internal).
- Peninjauan struktur kompensasi & benefit untuk level pendapatan rendah.
- Inisiatif *work-life balance* (kerja fleksibel, dukungan kesejahteraan).
- Mekanisme umpan balik & pemantauan berkala.
- Pendekatan *data-driven* untuk segmentasi & analisis lanjutan.
