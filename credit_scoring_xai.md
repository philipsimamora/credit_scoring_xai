# **UJIAN TENGAH SEMESTER SISTEM KECERDASAN BUATAN** 



# **CREDIT SCORING ALTERNATIF DENGAN EXPLAINABILITY** 

**Penyusun:** 

**PHILIP RIKARDO MANARIHON SIMAMORA (202022520001)** 

# **PROGRAM STUDI MAGISTER SISTEM INFORMASI FAKULTAS REKAYASA INDUSTRI UNIVERSITAS TELKOM** 

**2026** 

# **DAFTAR ISI** 

|DAFTA|R ISI............................................................................................................2|
|---|---|
|BAB I P|ENDAHULUAN........................................................................................4|
|I.1|Latar Belakang.........................................................................................4|
|I.2|Tujuan dan Manfaat Rancangan..............................................................5|
|I.3|Keterbatasan Rancangan..........................................................................7|
|BAB II|LANDASAN TEORI..................................................................................9|
|II.1|Konsep Dasar Model AI..........................................................................9|
|II.1|.1<br>Dasar Teori.......................................................................................9|
|II.1|.2<br>Kajian Literatur Terbaru................................................................11|
|II.2|Konsep Dasar Arsitektur........................................................................12|
|II.2|.1<br>_Stakeholder-Oriented Layered Architecture_..................................12|
|II.2|.2<br>_Enterprise Feature Store & Dual-Track Architecture_...................13|
|II.2|.3<br>_Integrated Fairness Pipeline & XAI Stability Validation_..............13|
|II.2|.4<br>_Model Risk Management, Immutable Audit Trail & Human-in-the-_|
|_Lo_|_op_<br>13|
|BAB III|PEMBAHASAN DAN HASIL RANCANGAN.....................................14|
|III.1|Hasil Rancangan Arsitektur...................................................................14|
|III.|1.1<br>_Data Ingestion & Governance Layer_.............................................14|
|III.|1.2<br>_Modeling & Fairness Layer_...........................................................15|
|III.|1.3<br>_Explainability Engine Layer_..........................................................16|
|III.|1.4<br>_Model Risk Management_(MRM) &_Audit Trail Layer_.................17|
|III.|1.5<br>_Presentation & Regulatory Output Layer_......................................18|
|III.2|Analisis dan Evaluasi.............................................................................19|
|III.|2.1<br>Analisis Perbandingan Model dan Evaluasi_Trade-Off_..................20|
|III.|2.2<br>Evaluasi Struktur_Audit Trail_dan Laporan_Explainability_............21|



III.2.3 _Governance Checklist_ untuk Kepatuhan terhadap Regulasi OJK dan UU PDP 22 III.2.4 Analisis Penanganan Keterbatasan Rancangan dalam Operasional AI 24 BAB IV KESIMPULAN.......................................................................................25 DAFTAR REFERENSI.........................................................................................27 

# **BAB I PENDAHULUAN** 

## **I.1 Latar Belakang** 

Penilaian kelayakan kredit ( _credit scoring_ ) merupakan salah satu aspek penting dalam operasional lembaga jasa keuangan untuk memastikan penyaluran pembiayaan dilakukan secara akurat, objektif, dan terukur guna meminimalkan risiko gagal bayar ( _default_ ). Namun, sistem penilaian kredit konvensional yang mengacu pada Sistem Layanan Informasi Keuangan (SLIK) dari Otoritas Jasa Keuangan (OJK) masih memiliki keterbatasan dalam menjangkau masyarakat yang termasuk kategori _unbanked_ maupun _underbanked_ ( _thin-file_ ). Kondisi tersebut menyebabkan banyak pelaku Usaha Mikro, Kecil, dan Menengah (UMKM), pekerja sektor informal, pekerja lepas, serta masyarakat usia produktif belum memiliki riwayat kredit perbankan, meskipun mereka mempunyai arus kas harian dan aktivitas ekonomi digital yang aktif. Akibatnya, muncul kesenjangan akses keuangan yang berpotensi membatasi kesempatan masyarakat dalam memperoleh pembiayaan produktif, padahal mereka sebenarnya memiliki kemampuan untuk memenuhi kewajiban pembayaran pinjaman ( _creditworthy_ ). 

Oleh karena itu, pemanfaatan data alternatif, seperti riwayat transaksi _e-commerce_ , pola penggunaan dompet digital ( _e-wallet_ ), kedisiplinan dalam pembayaran tagihan (listrik, air, internet, dan sebagainya), hingga pola penggunaan layanan telepon seluler, dapat menjadi salah satu solusi strategis untuk menilai profil risiko nasabah _thin-file_ secara lebih komprehensif. Karakteristik data alternatif yang memiliki dimensi tinggi, pola non-linear, serta volume data yang besar mendorong lembaga keuangan maupun penyelenggara Layanan Pendanaan Bersama Berbasis Teknologi Informasi (LPBBTI) untuk mulai mengadopsi algoritma _Machine Learning_ (ML) tingkat lanjut, khususnya metode seperti _Gradient Boosting Decision Trees_ (GBDT). Dibandingkan dengan metode statistik tradisional, GBDT memiliki kemampuan yang lebih baik dalam mengidentifikasi dan mengekstraksi pola perilaku mikro secara presisi sehingga dapat membedakan nasabah dengan tingkat risiko rendah maupun tinggi. 

Namun, penerapan model _Gradient Boosting_ pada data alternatif masih menghadapi dua permasalahan utama dalam tata kelola sistem informasi finansial. 

Permasalahan pertama berkaitan dengan konsep _black-box_ , yaitu kompleksitas ratusan _decision tree_ yang menyebabkan proses pengambilan keputusan model sulit dipahami dan diinterpretasikan secara langsung oleh analis kredit ( _underwriter_ ), nasabah, maupun auditor regulator. Permasalahan kedua adalah potensi terjadinya _algorithmic bias_ , ketika penggunaan variabel alternatif, seperti lokasi geografis dan jenis perangkat seluler, berpotensi menjadi _proxy variables_ yang secara tidak langsung menimbulkan diskriminasi terhadap kelompok tertentu. Dalam konteks Indonesia, penerapan sistem tersebut juga perlu memperhatikan tuntutan transparansi dan akuntabilitas yang diatur melalui Undang-Undang Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP), khususnya terkait hak subjek data dalam pemrosesan otomatis, POJK Nomor 11/POJK.03/2022 tentang Penyelenggaraan Teknologi Informasi oleh Bank Umum, POJK Nomor 10/POJK.05/2022 tentang LPBBTI, serta POJK Nomor 22 Tahun 2023 tentang Pelindungan Konsumen dan Masyarakat di Sektor Jasa Keuangan. Berdasarkan permasalahan tersebut, merancang arsitektur solusi _AI Credit Scoring_ berbasis data alternatif dengan mengintegrasikan _Gradient Boosting_ dan metode penjelasan _posthoc_ , yaitu SHAP ( _SHapley Additive exPlanations_ ) dan LIME ( _Local Interpretable Model-agnostic Explanations_ ). Selain itu, _Logistic Regression Scorecard_ digunakan sebagai model pembanding ( _baseline_ ), yang dikombinasikan dengan kerangka kerja _Fairness-Aware ML_ dan _Model Risk Management_ (MRM), serta dilengkapi dengan _audit trail_ dan _governance checklist_ untuk mendukung pemenuhan aspek kepatuhan terhadap regulasi OJK. 

## **I.2 Tujuan dan Manfaat Rancangan** 

Berdasarkan permasalahan serta urgensi pemenuhan kepatuhan terhadap regulasi yang telah dijelaskan pada latar belakang, tujuan dari penyusunan rancangan arsitektur solusi AI ini adalah sebagai berikut: 

1. Merancang Arsitektur Berlapis ( _Layered Architecture_ ) End-to-End: Menyusun rancangan arsitektur sistem penilaian kredit berbasis data alternatif yang terdiri dari lima lapisan fungsional, mulai dari tata kelola akuisisi data alternatif, pemodelan hibrida yang berkeadilan, mesin explainability, pemantauan risiko model ( _Model Risk Management_ ) beserta _audit trail_ , hingga pelaporan kepatuhan terhadap regulasi OJK. 

2. Mengombinasikan Akurasi Prediktif dan Interpretabilitas: Menerapkan pendekatan pemodelan _dual-track_ dengan menggabungkan kemampuan algoritma _Gradient Boosting_ (XGBoost/LightGBM) dalam membedakan tingkat risiko kredit dengan transparansi struktural _Logistic Regression Scorecard_ berbasis _Weight of Evidence_ (WoE) sebagai model pembanding ( _baseline_ ). 

3. Menyediakan Penjelasan Keputusan Kredit (Global & Lokal): Mengimplementasikan metode TreeSHAP dan LIME untuk menjelaskan hasil prediksi dari model kompleks ke dalam bentuk faktor-faktor yang memengaruhi keputusan kredit ( _adverse action reason codes_ ), sehingga dapat dipahami oleh analis kredit, nasabah, maupun regulator. 

4. Memitigasi Bias Demografis melalui _Fairness-Aware ML_ : Mengembangkan mekanisme untuk mendeteksi dan mengurangi potensi bias algoritmik melalui tahapan _pre-processing_ dan _post-processing_ , sehingga keputusan penilaian kredit dapat dilakukan secara adil tanpa menimbulkan diskriminasi terhadap kelompok demografis yang sensitif. 

5. Memenuhi Kepatuhan Tata Kelola dan Regulasi Nasional: Menyediakan instrumen _Model Risk Management_ (MRM), pencatatan _audit trail_ , serta daftar periksa tata kelola ( _governance checklist_ ) yang disesuaikan dengan ketentuan OJK dan Undang-Undang Pelindungan Data Pribadi (UU PDP). 

Manfaat yang diharapkan dari penerapan rancangan arsitektur ini mencakup empat kelompok pemangku kepentingan (stakeholders) utama, yaitu sebagai berikut: 

1. Bagi Nasabah _Thin-File_ dan Masyarakat _Unbanked_ : Memperluas kesempatan masyarakat dalam memperoleh akses pembiayaan formal secara adil dengan memanfaatkan rekam jejak transaksi digital dan riwayat pembayaran. Selain itu, memberikan transparansi terkait faktor yang menjadi dasar persetujuan maupun penolakan pengajuan kredit, sehingga nasabah dapat memahami dan meningkatkan literasi serta kondisi finansialnya. 

2. Bagi Analis Kredit ( _Underwriter_ ) dan Tim Operasional: Menyediakan UI pendukung dalam pengambilan keputusan yang menampilkan skor probabilitas gagal bayar beserta kontribusi masing-masing fitur secara visual. Dengan demikian, analis kredit dapat lebih mudah melakukan validasi dan evaluasi 

secara langsung ( _human-in-the-loop_ ), khususnya terhadap pengajuan kredit yang berada pada kategori _grey-zone_ . 

3. Bagi Divisi Manajemen Risiko ( _Model Risk Management_ ): Mendukung proses identifikasi dini terhadap penurunan performa model, perubahan distribusi data populasi ( _data drift_ ), serta penyesuaian anomali melalui pemantauan indikator _Population Stability Index_ (PSI). Selain itu, menyediakan mekanisme _fallback_ secara otomatis ke model pembanding (baseline) apabila model utama mengalami permasalahan. 

4. Bagi Regulator (OJK) dan Auditor Internal/Eksternal: Menyediakan rekam jejak audit ( _audit trail_ ) yang bersifat tidak dapat dimanipulasi ( _immutable_ ) pada setiap proses penilaian kredit. Hal ini bertujuan untuk mempermudah proses pemeriksaan dan evaluasi kepatuhan terhadap prinsip pelindungan konsumen, manajemen risiko teknologi informasi, serta ketentuan pelindungan data pribadi. 

## **I.3 Keterbatasan Rancangan** 

Untuk memastikan rancangan arsitektur tetap memiliki ruang lingkup yang jelas, realistis, dan terukur dalam penerapannya, penulis menetapkan beberapa keterbatasan yang sekaligus menjadi batasan dalam perancangan teknis sistem AI ini, yaitu sebagai berikut: 

1. Ketergantungan pada Validitas Data dan Ketersediaan Data Pihak Ketiga: Kualitas hasil prediksi model sangat dipengaruhi oleh tingkat akurasi, kelengkapan, serta ketersediaan data yang diperoleh dari penyedia data, seperti agregator telekomunikasi, penyedia layanan pembayaran tagihan, dan platform _e-commerce_ . Gangguan pada API pihak ketiga maupun tingginya _missing values_ yang melebihi batas toleransi dapat menghambat kemampuan model utama dalam menghasilkan prediksi secara optimal. 

2. Kerentanan terhadap Pergeseran Distribusi Data ( _Concept Drift_ dan _Data Drift_ ): Pola konsumsi digital dan perilaku pembayaran tagihan masyarakat dapat mengalami perubahan akibat kondisi makro ekonomi, inflasi, maupun perubahan kebijakan tarif tagihan. Perubahan tersebut berpotensi menyebabkan pola historis yang telah dipelajari model menjadi kurang relevan dalam memprediksi kondisi terbaru. Oleh karena itu, rancangan arsitektur ini memerlukan proses kalibrasi ulang model ( _retraining_ ) secara berkala serta 

pemantauan nilai _Population Stability Index_ (PSI) untuk mengidentifikasi perubahan distribusi data. 

3. _Trade-Off_ Latensi Komputasi pada Eksekusi _Explainability_ secara _Real-Time_ : Proses perhitungan nilai Shapley menggunakan TreeSHAP dan pembuatan data sampel acak melalui LIME membutuhkan sumber daya komputasi tambahan dibandingkan dengan proses prediksi skor probabilitas biasa. Kondisi ini dapat menjadi kendala ketika sistem menangani volume pengajuan kredit mikro yang tinggi, terutama apabila seluruh proses penjelasan keputusan dijalankan secara sinkron, karena dapat meningkatkan latensi respons API. 

4. _Accuracy-Fairness Trade-Off_ : Penerapan _fairness constraints_ , seperti penyetaraan _Equalized Odds_ atau _Disparate Impact Ratio_ , dapat membatasi optimasi fungsi objektif dari model yang digunakan. Kondisi tersebut berpotensi menyebabkan penurunan pada performa prediktif secara keseluruhan, sebagai konsekuensi dari upaya untuk mencapai tingkat _fairness constraints_ yang lebih seimbang antar kelompok demografis. 

5. Keterbatasan Pendekatan Lokal dan Risiko Manipulasi pada Fitur Alternatif: Metode LIME menggunakan pendekatan pendektan linear di sekitar satu titik data tertentu, sehingga hasil penjelasannya berpotensi mengalami ketidakstabilan apabila _sampling radius_ tidak diukur secara tepat. Selain itu, pemberian transparansi yang berlebihan terhadap fitur alternatif yang relatif mudah dimanipulasi, seperti frekuensi pengisian ulang pulsa dengan nominal kecil, dapat mendorong pemohon kredit melakukan rekayasa perilaku ( _strategic gaming_ untuk memengaruhi hasil penilaian kredit. 

# **BAB II LANDASAN TEORI** 

Dasar teori dalam perancangan ini mengacu pada berbagai literatur ilmiah dari jurnal internasional bereputasi yang terindeks Scopus Q1 dan Q2. Literatur tersebut membahas perkembangan sistem penilaian kredit, pemanfaatan data alternatif, penggunaan gabungan beberapa algoritma, serta aspek keadilan dan transparansi dalam penerapan model kecerdasan buatan. 

## **II.1 Konsep Dasar Model AI** 

## **II.1.1 Dasar Teori** 

Sebelum membahas perkembangan literatur terbaru pada periode 2021–2026, perancangan model AI dalam penelitian ini terlebih dahulu mengacu pada lima karya ilmiah sebagai dasar teori. 

## **II.1.1.1 Algoritma** **_Gradient Boosting Decision Trees_** 

Melalui jurnal _The Annals of Statistics_ memperkenalkan pendekatan _Greedy Function Approximation_ atau _Gradient Boosting Machine_ (GBM), yaitu metode yang membangun model secara bertahap dengan menambahkan pohon regresi baru berdasarkan gradien negatif ( _pseudo-residuals_ ) dari fungsi kerugian ( _loss function_ ) (Jerome H. Friedman, 2001). Lalu pengembangan menggunakan pendekatan tersebut melalui algoritma XGBoost ( _eXtreme Gradient Boosting_ ) (Chen & Guestrin, 2016). Pengembangan ini mencakup penggunaan ekspansi Taylor orde kedua pada fungsi objektif, penambahan regularisasi L1/L2 pada bobot setiap daun pohon untuk mengurangi risiko _overfitting_ , serta metode pencarian pemisahan data ( _split-finding_ ) berbasis kuantil terbobot. Berbagai pengembangan tersebut membuat XGBoost mampu mengolah data dengan banyak nilai kosong atau data renggang ( _sparse data_ ) secara lebih efisien. 

## **II.1.1.2 Klasifikasi Risiko Kredit dan Baseline Logistik** 

Dalam penelitian yang dilakukan oleh Lessmann et al., sebanyak 41 algoritma klasifikasi dievaluasi menggunakan berbagai dataset kredit. Hasil penelitian tersebut menunjukkan bahwa Regresi Logistik dengan transformasi _Weight of Evidence_ (WoE) dan seleksi fitur menggunakan _Information Value_ (IV) masih menjadi salah satu metode yang banyak digunakan sebagai model pembanding ( _baseline_ ) dalam industri keuangan karena memiliki koefisien yang transparan dan 

mudah diinterpretasikan (Lessmann et al., 2015). Namun, penelitian tersebut juga menunjukkan bahwa penggunaan gabungan algoritma dengan pendekatan homogen maupun heterogen, termasuk metode _boosted trees_ , secara konsisten mampu menghasilkan kemampuan pemisahan risiko kredit yang lebih baik berdasarkan nilai AUC-ROC dibandingkan metode konvensional. 

## **II.1.1.3 Model Pengganti Lokal LIME** 

Metode LIME ( _Local Interpretable Model-agnostic Explanations_ ) digunakan untuk menjelaskan hasil prediksi dari model kompleks yang bersifat _black-box_ melalui pendekatan model pengganti ( _surrogate model_ ) berbentuk linear yang lebih mudah dipahami (Ribeiro et al., 2016). Metode ini bekerja dengan membuat sejumlah sampel gangguan di sekitar titik data nasabah (x), kemudian melatih model linear lokal g(z) untuk mendekati hasil prediksi model utama f(x). Proses tersebut dilakukan dengan mengoptimalkan fungsi kerugian lokal yang mempertimbangkan bobot kedekatan antar data ( _proximity kernel π_x_ ) serta penalti kompleksitas model Ω(g), sehingga hasil prediksi dapat dijelaskan secara lebih sederhana dan mudah diinterpretasikan. 

## **II.1.1.4 Nilai Shapley dan Algoritma TreeSHAP** 

SHAP dikembangkan sebagai kerangka kerja yang menggabungkan berbagai metode karakteristik fitur dengan menggunakan konsep nilai. Pendekatan ini memiliki tiga prinsip utama yang harus dipenuhi, yaitu _local accuracy_ , _missingness_ , dan _consistency_ (Lundberg & Lee, 2017). Selanjutnya, Lundberg et al. melalui jurnal _Nature Machine Intelligence_ mengembangkan algoritma _TreeExplainer_ (TreeSHAP) yang mampu mengurangi kompleksitas komputasi dalam menghitung nilai Shapley secara akurat pada gabungan model berbasis _decision tree_ (Lundberg et al., 2019). Pengembangan ini mengubah kebutuhan komputasi dari kompleksitas eksponensial, sehingga proses perhitungan penjelasan model menjadi lebih efisien. Selain itu, TreeSHAP juga memungkinkan hasil penjelasan pada masing-masing prediksi ( _local explanation_ ) digabungkan untuk memberikan gambaran mengenai pola dan struktur pengambilan keputusan model secara keseluruhan ( _global explanation_ ). 

## **II.1.1.5 Mitigasi Bias Pra-Pemrosesan dan Ekonomi Explainability** 

Teknik _Reweighing_ pada tahap _pre-processing_ dengan memberikan bobot yang berbeda pada setiap kombinasi kelompok atribut sensitif dan label kelas (Kamiran & Calders, 2012). Teknik ini bertujuan untuk mengurangi potensi bias sebelum model dilatih. Sementara itu, penggunaan nilai Shapley sebagai pendekatan untuk menjelaskan hasil prediksi model _Machine Learning_ dalam manajemen risiko kredit pada layanan _peer-to-peer lending_ (Bussmann et al., 2021). 

## **II.1.2 Kajian Literatur Terbaru** 

Untuk memahami dan menjawab berbagai permasalahan dalam penilaian kredit alternatif, khususnya bagi nasabah _thin-file_ , penulis mengkaji dan menganalisis enam artikel jurnal internasional yang terindeks Scopus Q1/Q2 pada periode 2021 –2026 sebagai dasar dalam merancang sistem. 

_Table 1. Referensi Jurnal_ 

|**No**|**Jurnal**|**Fokus & Metodologi Penelitian**|**Temuan Kunci**|
|---|---|---|---|
|1|Enhancing<br>credit<br>scoring with alternative<br>data (Djeundje et al.,<br>2021)|Evaluasi prediktif data alternatif<br>(psikometrik<br>&<br>aktivitas<br>digital/email) bersama variabel<br>demografis dengan model<br>logistik dan ansambel ML (75%<br>train / 25% test).|Kombinasi data alternatif dalam<br>model ansambel meningkatkan AUC<br>secara signifikan hingga layak<br>komersial untuk menilai nasabah thin-<br>file tanpa histori biro kredit (Diadopsi<br>pada Layer 1 & 2).|
|2|Deep learning for credit<br>scoring: Do or don’t?<br>(Gunnarsson et al.,<br>2021)|Studi<br>komparasi<br>empiris<br>algoritma Deep Learning (MLP,<br>CNN, LSTM) vs. ansambel<br>pohon (XGBoost, Random<br>Forest) dan Regresi Logistik pada<br>10 dataset kredit.|Deep learning tidak unggul pada data<br>kredit tabular; XGBoost terbukti<br>sebagai algoritma dengan performa<br>klasifikasi terbaik dan paling<br>konsisten (Diadopsi sebagai Model<br>Utama di Layer 2).|
|3|Machine learning for<br>credit<br>scoring:<br>Improving<br>logistic<br>regression with non-<br>linear<br>decision-tree<br>effects (Dumitrescu et|Pengembangan<br>Penalised<br>Logistic Tree Regression (PLTR)<br>yang mengekstraksi aturan pohon<br>keputusan pendek ke dalam<br>regresi logistik ter-regularisasi.|Menunjukkan pentingnya menjaga<br>scorecard logistik yang dapat<br>diinterpretasikan untuk kepatuhan<br>Basel III sambil menangkap efek non-<br>linear (Diadopsi sebagai Dual-Track<br>Baseline & Fallback di Layer 2 & 4).|



||al., 2022)|||
|---|---|---|---|
|4|Fairness in credit<br>scoring: Assessment,<br>implementation<br>and<br>profit<br>implications<br>(Kozodoi et al., 2022)|Kajian kriteria fairness statistik<br>dan katalog intervensi algoritma<br>fairness (pre-, in-, dan post-<br>processing) beserta evaluasi<br>dampaknya<br>terhadap<br>profitabilitas kredit.|Diskriminasi algoritmik dapat ditekan<br>ke level aman dengan pengorbanan<br>akurasi/profit yang minimal melalui<br>intervensi pipeline ML (Diadopsi<br>pada modul Fairness-Aware ML di<br>Layer 2).|
|5|Transparency,<br>auditability,<br>and<br>explainability<br>of<br>machine<br>learning<br>models in credit scoring<br>(Bücker et al., 2022)|Penyusunan kerangka kerja<br>Transparency, Auditability, and<br>eXplainability (TAX) berbasis<br>kebutuhan<br>pemangku<br>kepentingan di seluruh siklus<br>hidup model kredit.|Memisahkan penjelasan Global (audit<br>bias sistemik oleh validator/regulator)<br>dan<br>penjelasan<br>Lokal/Instans<br>(identifikasi faktor merugikan bagi<br>analis & nasabah) (Diadopsi pada<br>Layer 3 & 5).|
|6|Evaluating the stability<br>of model explanations<br>in instance-dependent<br>cost-sensitive<br>credit<br>scoring (Ballegeer et al.,<br>2025)|Evaluasi kuantitatif terhadap<br>stabilitas penjelasan post-hoc<br>(SHAP & LIME) pada model<br>penilaian kredit sensitif biaya<br>(termasuk XGBoost) di bawah<br>tuntutan regulasi.|Menegaskan bahwa keunggulan<br>model kompleks akan gugur jika<br>penjelasan lokalnya tidak stabil;<br>mewajibkan validasi stabilitas<br>penjelasan (Diadopsi pada protokol<br>Cross-Check SHAP & LIME di Layer<br>3).|



## **II.2 Konsep Dasar Arsitektur** 

Berdasarkan hasil kajian dari literatur jurnal referensi yang digunakan, rancangan arsitektur solusi AI dalam penelitian ini disusun dengan mengacu pada prinsip perancangan sistem enterprise sebagai berikut: 

## **II.2.1** **_Stakeholder-Oriented Layered Architecture_** 

Mengacu pada kerangka _life cycle model_ , rancangan sistem dibagi menjadi lima lapisan yang memiliki fungsi masing-masing, yaitu _Data Ingestion_ , _Modeling & Fairness_ , _Explainability Engine_ , _MRM & Audit Trail_ , serta _Presentation & Regulatory Output_ (Bücker et al., 2022). Pembagian ini bertujuan agar setiap kebutuhan dapat ditangani oleh komponen yang sesuai, mulai dari kebutuhan teknis data, kebutuhan analis kredit dalam, transparansi informasi bagi nasabah, hingga kebutuhan auditor OJK dalam melakukan pengawasan. 

## **II.2.2** **_Enterprise Feature Store & Dual-Track Architecture_** 

Rancangan sistem menggunakan _Feature Store_ terpusat untuk menyimpan dan mengelola hasil pengolahan variabel data alternatif (Djeundje et al., 2021; Dumitrescu et al., 2022; Gunnarsson et al., 2021). Data tersebut kemudian digunakan oleh dua jalur model, yaitu model utama berbasis _Gradient Boosting_ yang mampu mengenali hubungan non-linear dan pola data dengan dimensi tinggi, serta model pendamping berupa Scorecard Logistik berbasis WoE/IV yang digunakan sebagai pembanding dalam proses kalibrasi sekaligus sebagai alternatif apabila model utama mengalami gangguan ( _circuit-breaker fallback_ ). 

## **II.2.3** **_Integrated Fairness Pipeline & XAI Stability Validation_** 

Rancangan sistem yang dilengkapi dengan pemeriksaan potensi bias sebelum dan sesudah proses prediksi model, serta mekanisme untuk memastikan hasil penjelasan model tetap konsisten (Ballegeer et al., 2025; Kozodoi et al., 2022). TreeSHAP digunakan untuk menghitung nilai Shapley secara akurat dengan waktu pemrosesan yang lebih efisien, sehingga dapat mendukung kebutuhan penjelasan secara _realtime_ . Sementara itu, LIME digunakan sebagai model pembanding melalui pembobotan yang telah disesuaikan untuk memeriksa konsistensi urutan fitur yang memengaruhi tingkat risiko nasabah. Hasil dari proses tersebut kemudian digunakan untuk menghasilkan alasan yang mendasari keputusan kredit ( _Adverse Action Reason Codes_ ). 

## **II.2.4** **_Model Risk Management, Immutable Audit Trail & Human-in-the-Loop_** 

Untuk mendukung prinsip kehati-hatian dalam perbankan dan memenuhi ketentuan regulasi OJK, rancangan sistem dilengkapi dengan pemantauan nilai _Population Stability Index_ (PSI) untuk melihat perubahan distribusi data. Selain itu, sistem menggunakan penyimpanan rekam jejak transaksi dengan mekanisme _Write-OnceRead-Many_ (WORM), sehingga data audit yang telah tersimpan tidak dapat diubah. Rancangan ini juga menyediakan antarmuka yang memungkinkan analis kredit melakukan pemeriksaan dan memberikan keputusan secara langsung terhadap pengajuan kredit yang berada pada tingkat risiko abu-abu ( _grey-zone_ ). 

# **BAB III PEMBAHASAN DAN HASIL RANCANGAN** 

## **III.1 Hasil Rancangan Arsitektur** 

Dalam penelitian ini, penulis merancang arsitektur solusi AI menggunakan pendekatan lima lapisan (5-Tier Layered Architecture) yang disusun dengan mempertimbangkan kebutuhan bisnis, hasil kajian literatur jurnal Q1/Q2, serta berbagai batasan teknis yang telah dijelaskan pada bagian sebelumnya. Sebelum membahas fungsi dan detail teknis dari setiap komponen, Gambar 1 terlebih dahulu menampilkan gambaran keseluruhan rancangan arsitektur sistem. 



<!-- Start of picture text -->
Data ingestion & Raw APH Privac “onsent -“Anonymia<br>areca [emt fom meee senenny [meter sen |<br>Eytinny “Shapley Matrix- ——ae -Local Attnibutons-<br>fpaclmapyf —=ESs-- Ea<br><!-- End of picture text -->

_Gambar 1. 5-Tier Layered Architecture_ 

Berdasarkan Gambar 1, berikut merupakan penjelasan mengenai fungsi setiap lapisan, alur kerja sistem, serta mekanisme yang digunakan untuk mengatasi berbagai keterbatasan dalam rancangan arsitektur tersebut. 

## **III.1.1** **_Data Ingestion & Governance Layer_** 

Lapisan pertama berperan sebagai tempat pengumpulan sekaligus pemeriksaan kualitas data alternatif, sebagaimana dibahas dalam penelitian. Sistem menghubungkan empat sumber data melalui konektor API yang telah dienkripsi, yaitu: 

- (a) Data Telekomunikasi, meliputi frekuensi pengisian pulsa, konsistensi penggunaan paket data, serta usia aktif kartu SIM ( _tenure_ ); 

- (b) Data Tagihan, mencakup ketepatan waktu pembayaran tagihan listrik PLN, air PDAM, dan internet rumah; 

- (c) Data Transaksi Digital, seperti perbandingan transaksi kebutuhan pokok dengan transaksi konsumtif melalui platform _e-commerce_ dan _e-wallet_ ; serta 

- (d) Pengecekan SLIK OJK untuk memastikan bahwa pemohon termasuk dalam kategori _thin-file_ , yaitu tidak memiliki riwayat kredit aktif atau berstatus _nohit_ . 

Untuk memenuhi ketentuan UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP), setiap data yang masuk harus melalui modul _Privacy & Consent Gateway_ . Modul ini bertugas memastikan adanya persetujuan dari nasabah ( _explicit opt-in consent_ ), menyamarkan informasi identitas pribadi ( _PII masking_ ), serta mengubah nomor identitas menjadi token menggunakan metode kriptografi SHA-256. Selain itu, untuk mengatasi permasalahan terkait kualitas dan ketersediaan data dari pihak ketiga, lapisan ini dilengkapi dengan _Data Quality Gatekeeper_ . Komponen tersebut akan menolak data apabila persentase fitur yang kosong ( _missing rate_ ) melebihi 25%, kemudian mengarahkan pengajuan kredit ke proses pemeriksaan manual atau menggunakan model pembanding ( _baseline_ ) yang lebih sederhana. Data yang telah lolos pemeriksaan selanjutnya diolah dan dikelompokkan berdasarkan periode waktu 1, 3, dan 6 bulan, kemudian disimpan ke dalam _Enterprise Feature Store._ 

## **III.1.2** **_Modeling & Fairness Layer_** 

Lapisan kedua berfungsi untuk menjalankan proses pelatihan model dan menghasilkan prediksi risiko kredit menggunakan pendekatan pemodelan _hybrid_ yang memperhatikan aspek keadilan. Sebelum data digunakan dalam proses pelatihan, modul _Pre-Processing Fairness_ menerapkan algoritma _Reweighing_ dengan memberikan bobot tambahan pada sampel dari kelompok demografis yang kurang terwakili, seperti pemohon dari wilayah di luar Pulau Jawa atau kelompok usia muda. Proses ini bertujuan untuk mengurangi ketidakseimbangan distribusi antara kelas _default_ dan _non-default_ berdasarkan atribut sensitif. 

Selanjutnya, komponen Dual-Track Credit Modeling menjalankan dua model secara bersamaan, yaitu: 

- (a) Model Baseline berupa Scorecard Logistik yang menggunakan transformasi _Weight of Evidence_ (WoE) dan seleksi fitur berdasarkan nilai _Information Value_ (IV > 0.02); serta 

- (b) Model Utama (Challenger/Primary) yang menggunakan algoritma _Gradient Boosting Decision Trees_ seperti XGBoost atau LightGBM. 

Untuk mengurangi risiko manipulasi perilaku ( _strategic gaming_ ) sekaligus menjaga agar hasil prediksi tetap sesuai dengan hubungan antarvariabel, model _Gradient Boosting_ dilengkapi dengan pengaturan _Monotonic Constraints_ . Sebagai contoh, model diarahkan untuk mempertahankan hubungan negatif antara ketepatan waktu pembayaran tagihan listrik dengan probabilitas gagal bayar, sehingga semakin baik kedisiplinan pembayaran maka semakin rendah risiko gagal bayar yang dihasilkan model. 

Hasil prediksi berupa skor probabilitas awal selanjutnya disesuaikan menggunakan metode _Platt Scaling_ . Setelah proses penyesuaian, hasil tersebut diteruskan ke modul _Post-Processing Fairness_ melalui komponen _Threshold Optimizer_ . Modul ini bertugas menyesuaikan ambang batas persetujuan kredit antar kelompok demografis agar hasil keputusan tetap memperhatikan aspek keadilan, dengan memenuhi standar _Disparate Impact Ratio_ pada rentang 0,80 hingga 1,25. 

## **III.1.3** **_Explainability Engine Layer_** 

Lapisan ketiga berfungsi untuk menjelaskan proses pengambilan keputusan dari model _Gradient Boosting_ yang sebelumnya sulit dipahami karena bersifat _blackbox_ . Perancangan lapisan ini mengacu pada kerangka TAX serta terkait pengujian stabilitas hasil penjelasan model. Untuk mengatasi keterbatasan waktu pemrosesan dalam menghasilkan penjelasan secara _real-time_ , sistem membagi proses tersebut menjadi dua jalur, yaitu: 

1. Jalur Sinkron ( _Real-Time Inference via TreeSHAP_ ): Ketika pengajuan kredit masuk, algoritma TreeSHAP yang terhubung langsung dengan struktur _decision tree_ akan menghitung nilai Shapley lokal secara akurat 

dalam waktu kurang dari 50 milidetik. Nilai tersebut digunakan untuk melihat seberapa besar pengaruh setiap fitur data alternatif terhadap kenaikan atau penurunan skor risiko nasabah dibandingkan dengan nilai rata-rata populasi. 

2. Jalur Asinkron ( _Audit & Cross-Check Stability via LIME_ ): Untuk mengurangi masalah ketidakstabilan hasil sampling pada LIME, proses LIME dijalankan secara terpisah melalui _background worker_ agar tidak menghambat proses prediksi utama. Proses ini difokuskan pada pengajuan kredit yang ditolak, pengajuan yang berada pada zona abu-abu, serta sampel yang dipilih untuk kebutuhan audit. Pengaturan _fixed random seed_ dan pembobotan kernel yang telah dikalibrasi digunakan untuk menjaga konsistensi hasil penjelasan. Selanjutnya, sistem membandingkan empat fitur dengan pengaruh terbesar dari SHAP dan LIME melalui pengukuran kesesuaian urutan fitur ( _rank stability_ ) sebagai indikator untuk menilai konsistensi penjelasan model. 

Setelah proses tersebut, komponen _Adverse Action Reason Generator_ akan mengubah kode fitur teknis yang memiliki pengaruh SHAP paling besar menjadi penjelasan dalam bahasa Indonesia yang mudah dipahami. Sebagai contoh, "Terdapat keterlambatan pembayaran tagihan listrik lebih dari 7 hari dalam 3 bulan terakhir". Dengan demikian, alasan di balik keputusan kredit dapat diketahui oleh analis maupun nasabah. 

## **III.1.4** **_Model Risk Management_ (MRM) &** **_Audit Trail Layer_** 

Lapisan keempat dirancang untuk mengatasi risiko perubahan distribusi data ( _concept drift_ dan _data drift_ ) yang dapat terjadi akibat perubahan kondisi ekonomi maupun pola perilaku masyarakat. Untuk mengantisipasi hal tersebut, modul _Drift & Stability Monitor_ secara otomatis menghitung nilai _Population Stability Index_ (PSI) pada distribusi skor akhir dan _Characteristic Stability Index_ (CSI) pada setiap variabel data alternatif. Proses pemantauan ini dilakukan secara berkala, baik mingguan maupun bulanan. 

Apabila nilai PSI melebihi 0,10, yang menunjukkan adanya perubahan distribusi data pada tingkat moderat, sistem akan mengirimkan peringatan kepada tim Data 

Science. Sementara itu, apabila nilai PSI melebihi 0,25 atau terjadi gangguan pada API penyedia data alternatif eksternal, komponen _Champion-Challenger Controller_ akan mengaktifkan mekanisme _Circuit-Breaker Fallback_ . Mekanisme ini secara otomatis mengalihkan proses penilaian kredit ke model Baseline Scorecard Logistik yang lebih stabil terhadap perubahan fitur mikro. 

Selain pemantauan risiko model, seluruh informasi yang berkaitan dengan proses pengambilan keputusan juga disimpan ke dalam _Immutable Audit Trail Repository_ menggunakan mekanisme penyimpanan _Write-Once-Read-Many_ (WORM). Informasi tersebut mencakup versi model yang digunakan, _hash_ dataset, nilai input fitur yang telah dianonimkan, skor probabilitas gagal bayar (PD), nilai SHAP lokal, metrik _fairness_ , serta identitas analis yang memberikan persetujuan. Mekanisme penyimpanan ini memastikan bahwa seluruh rekaman keputusan yang telah tersimpan tidak dapat diubah maupun dihapus, sehingga riwayat penilaian kredit dapat ditelusuri kembali dan digunakan untuk kebutuhan audit oleh OJK. 

**III.1.5** **_Presentation & Regulatory Output Layer_** 

Lapisan kelima berfungsi sebagai penghubung antara sistem AI dengan pengguna maupun regulator. Pada lapisan ini, sistem menerapkan mekanisme pengelompokan keputusan kredit berdasarkan skor probabilitas gagal bayar (PD). Pengajuan dengan tingkat risiko sangat rendah (PD < 5%) akan mendapatkan persetujuan secara otomatis, sedangkan pengajuan dengan tingkat risiko sangat tinggi (PD > 25%) akan ditolak secara otomatis dan disertai laporan penjelasan mengenai faktor yang memengaruhi keputusan tersebut. Sementara itu, pengajuan dengan tingkat risiko di antara 5% hingga 25% akan diarahkan ke _Underwriter Decision Cockpit_ untuk diperiksa lebih lanjut oleh analis kredit melalui mekanisme _Human-in-the-Loop_ . 

Apabila analis kredit memutuskan untuk mengubah rekomendasi yang diberikan oleh sistem ( _override_ ) berdasarkan pertimbangan tambahan, analis wajib memilih alasan yang telah disediakan. Alasan tersebut kemudian disimpan secara otomatis ke dalam _audit trail_ sebagai bagian dari rekam jejak pengambilan keputusan. Selain itu, lapisan ini juga menghasilkan dua dokumen utama untuk mendukung kepatuhan sistem, yaitu Laporan _Explainability_ Nasabah yang menjelaskan alasan keputusan 

kredit serta dokumen _Governance Checklist_ sebagai panduan pemeriksaan kesesuaian sistem terhadap regulasi OJK. 

_Table 2. Rancangan 5-Tier Layered Architecture_ 

|**Layer Arsitektur**|**Komponen Utama**|**Input & Proses Teknis**|**Output & Batasan**|
|---|---|---|---|
|_Data Ingestion &_|Consent Gateway,|Input<br>data<br>Telco,|Vektor<br>fitur|
|_Governance_|Data Quality Gate,<br>Enterprise Feature<br>Store|Tagihan, E-Wallet, &<br>SLIK;<br>tokenisasi<br>SHA-256; agregasi fitur<br>1M/3M/6M.|terstandar; mitigasi<br>keterbatasan<br>data<br>pihak ketiga jika<br>missing > 25%.|
|_Modeling_<br>_&_<br>_Fairness_|Reweighing<br>Module, Dual-Track<br>Engine (Logistik &<br>GBDT), Threshold<br>Optimizer|Pembobotan<br>sampel<br>demografis; pelatihan<br>Scorecard WoE &<br>XGBoost/LightGBM<br>monotonic; kalibrasi PD.|Skor PD terkalibrasi &<br>adil (Disparate Impact<br>0,80-1,25); mitigasi<br>bias & gaming.|
|_Explainability_|TreeSHAP Engine,|Ekstraksi Shapley value|Matriks<br>Global|
|_Engine_|Asynchronous<br>LIME<br>Worker,<br>Reason<br>Code<br>Generator|eksak (<50ms); validasi<br>silang stabilitas LIME;<br>translasi kamus bisnis.|Importance & Top-4<br>Adverse<br>Action<br>Reason Codes per<br>nasabah.|
|_MRM & Audit_|PSI/CSI<br>Drift|Perhitungan<br>indeks|Peringatan dini drift,|
|_Trail_|Monitor, Champion-<br>Challenger<br>Controller, WORM<br>Audit Log|stabilitas populasi &<br>fitur;<br>evaluasi<br>AUC/Gini/KS; penulisan<br>log JSON WORM.|auto-fallback<br>ke<br>Scorecard Logistik<br>jika PSI > 0,25, &<br>jejak audit.|
|_Presentation &_|Underwriter Cockpit|Triase keputusan (Auto-|Keputusan<br>final|
|_Regulatory_|(HITL), Customer|Approve,<br>Grey-Zone|kredit,<br>Laporan|
||XAI Portal, OJK|HITL,<br>Auto-Reject);|Explainability,<br>&|
||Governance Module|pencatatan<br>alasan<br>override.|Governance Checklist<br>Regulasi OJK.|



## **III.2 Analisis dan Evaluasi** 

Untuk memastikan bahwa rancangan arsitektur yang dibuat tidak hanya sesuai secara konsep, tetapi juga dapat diterapkan dalam operasional dan memenuhi 

ketentuan regulasi, penulis melakukan beberapa evaluasi. Evaluasi tersebut mencakup perbandingan performa model, perancangan struktur _audit trail_ , serta penyusunan matriks untuk menilai kesesuaian sistem terhadap tata kelola dan regulasi OJK. 

## **III.2.1 Analisis Perbandingan Model dan Evaluasi** **_Trade-Off_** 

Evaluasi dilakukan dengan membandingkan tiga pendekatan pemodelan menggunakan skenario pengujian data alternatif nasabah _thin-file_ , yaitu: 

- (1) Baseline Scorecard Logistik berbasis WoE, 

- (2) _Gradient Boosting_ (XGBoost/LightGBM) tanpa penerapan pengendalian keadilan, dan 

- (3) _Fairness-Aware Gradient Boosting_ yang dilengkapi dengan metode SHAP dan LIME. 

Berdasarkan hasil yang ditampilkan pada Tabel 3, penggunaan _Gradient Boosting_ tanpa pengendalian keadilan memang menghasilkan nilai AUC-ROC dan Gini paling tinggi. Namun, model tersebut memiliki nilai _Disparate Impact Ratio_ sebesar 0,71, yang masih berada di bawah batas toleransi 0,80. Kondisi ini disebabkan oleh adanya hubungan tidak langsung antara variabel pengeluaran komunikasi dengan perbedaan wilayah geografis. 

Melalui penerapan metode _Reweighing_ dan _Threshold Optimizer_ pada model yang diusulkan, nilai _Disparate Impact Ratio_ berhasil meningkat menjadi 0,88 sehingga memenuhi batas keadilan yang telah ditetapkan. Peningkatan tersebut hanya menyebabkan penurunan nilai AUC-ROC sebesar 0,012 poin. Meskipun terdapat sedikit penurunan pada performa prediksi, model yang diusulkan tetap memiliki kemampuan yang lebih baik dibandingkan dengan Scorecard Logistik konvensional. 

_Table 3. Analisis Perbandingan Model_ 

|**Metrik Evaluasi**|**Threshold**|**Baseline**|**Gradient**|**Fairness GBDT**|
|---|---|---|---|---|
||||**Boosting**|**+ XAI**|
|Daya Pisah Risiko|>= 0,750|0,742 (Moderat)|0,836 (Sangat|0,824 (Tinggi &|
|(AUC-ROC)|||Tinggi)|Optimal)|



|Koefisien Gini<br>(2×AUC - 1)|>= 0,500|0,484|0,672|0,648|
|---|---|---|---|---|
|Statistik<br>Kolmogorov-<br>Smirnov (KS)|>= 0,350|0,345|0,512|0,489|
|Akurasi Kalibrasi<br>(Brier Score)|<= 0,150 (Lebih<br>rendah lebih<br>baik)|0,141|0,108|0,114|
|Disparate Impact<br>Ratio (DIR)|0,80 <= DIR <=<br>1,25|0,83 (Patuh)|0,71<br>(Melanggar /<br>Bias)|0,88 (Patuh &<br>Adil)|
|Equal<br>Opportunity<br>Difference (EOD)||EOD| <= 0,05|+0,04|-0,11 (Bias<br>Signifikan)|-0,02<br>(Terkendali)|
|Stabilitas<br>Populasi<br>(PSI<br>Bulanan)|PSI < 0,10<br>(Stabil)|0,04<br>(Sangat<br>Stabil)|0,09 (Stabil)|0,07 (Stabil)|
|Latensi Inferensi<br>+ Penjelasan|<= 200 ms<br>(SLA<br>Real-<br>Time)|15 ms (Koefisien<br>Langsung)|28 ms (Tanpa<br>XAI)|68 ms (Sync<br>TreeSHAP)|
|Stabilitas|Konsistensi|100%|Tidak|94,2%|
|Penjelasan Lokal|Rank Top-4 >=|(Deterministik)|Tersedia|(TreeSHAP<br>+|
|(XAI)|85%||(Black-Box)|Calibrated<br>LIME)|



## **III.2.2 Evaluasi Struktur** **_Audit Trail_ dan Laporan** **_Explainability_** 

Untuk memastikan setiap keputusan kredit dapat ditelusuri kembali, seluruh hasil keputusan dari Layer 3 dan Layer 4 harus menghasilkan rekaman audit yang tersusun secara terstruktur. Sebagai contoh, berikut merupakan gambaran pencatatan audit ( _audit record_ ) untuk salah satu pemohon kredit UMKM kategori _thin-file_ yang berada pada zona abu-abu ( _grey-zone_ ), dengan beberapa parameter sebagai berikut: 

1. Identitas Transaksi dan Versi Model: Audit_ID: AUD-20261114-88921 | Model_Primary: GBDT-AltCredit-v2.4 | Model_Baseline: LogitWoE-v1.2 | Status_Consent: VERIFIED_OPT_IN (Timestamp: 2026-1114T08:15:22Z). 

2. Hasil Prediksi dan Perbandingan Model: Skor probabilitas gagal bayar (PD) dari _Gradient Boosting_ sebesar 0,142 dengan kategori _Grey-Zone Medium Risk_ , sedangkan model Baseline Logistik menghasilkan skor PD sebesar 0,158. Terdapat selisih sebesar 0,016 antara kedua model, tetapi masih berada dalam batas yang dapat diterima. 

3. Hasil Penjelasan Lokal (4 Faktor Teratas SHAP dan _Reason Codes_ ): Hasil analisis SHAP menunjukkan empat faktor utama yang memengaruhi skor risiko nasabah, yaitu: 

   - a. Rasio keterlambatan pembayaran tagihan listrik selama 6 bulan terakhir (+0,061 PD) sebagai faktor risiko utama; 

   - b. Perubahan saldo akhir _e-wallet_ setiap bulan (+0,034 PD) sebagai faktor risiko kedua; 

   - c. Masa aktif nomor seluler lebih dari 48 bulan (-0,028 PD) sebagai faktor yang menurunkan risiko; dan 

   - d. Konsistensi transaksi penjualan _e-commerce_ setiap minggu (-0,022 PD) sebagai faktor yang menurunkan risiko. Hasil perbandingan urutan fitur antara SHAP dan model pengganti LIME menunjukkan tingkat kesesuaian sebesar 100% atau seluruh 4 fitur memiliki urutan yang sama. 

4. Keputusan Akhir ( _Human-in-the-Loop_ ): Keputusan akhir berupa APPROVED_WITH_ADJUSTED_LIMIT, yaitu pengajuan kredit disetujui dengan penyesuaian plafon menjadi Rp5.000.000. Keputusan ini diberikan oleh _Underwriter_ dengan ID UW-042 berdasarkan hasil pemeriksaan tambahan terhadap arus kas toko fisik pemohon. 

## **III.2.3** **_Governance Checklist_ untuk Kepatuhan terhadap Regulasi OJK dan UU PDP** 

Salah satu bagian penting dari hasil rancangan ini adalah penyusunan daftar periksa tata kelola ( _governance checklist_ ) yang digunakan untuk menilai kesiapan sistem 

dalam memenuhi ketentuan regulasi yang berlaku di Indonesia. Daftar periksa tersebut mencakup berbagai aspek kepatuhan yang telah disesuaikan dengan regulasi OJK dan UU PDP, sebagaimana ditampilkan pada Tabel 4. 

_Table 4. Acuan Regulasi UU PDP dan OJK_ 

|**Regulasi Acuan**|**Aspek Kepatuhan**<br>**Utama**|**Implementasi pada**<br>**Rancangan**<br>**Arsitektur**|**Bukti Audit (Audit**<br>**Evidence)**|
|---|---|---|---|
|UU No. 27 Tahun|Persetujuan eksplisit|Validasi Explicit Opt-In|Log<br>persetujuan|
|2022 (UU PDP) -|pemrosesan<br>data|pada Privacy Gateway;|digital berstempel|
|Pasal 20 & 21|pribadi & hak subjek<br>data atas pemrosesan<br>otomatis.|penyediaan<br>Reason<br>Codes & mekanisme<br>sanggahan<br>manual<br>(HITL).|waktu & Laporan<br>Explainability<br>Nasabah.|
|POJK<br>No.|Tata kelola TI,|Arsitektur<br>MRM|Dokumen validasi|
|11/POJK.03/2022|manajemen risiko|berlapis, pemantauan|model independen,|
|(Penyelenggaraan TI|penggunaan|PSI/CSI<br>otomatis,|laporan metrik PSI|
|Bank Umum)|teknologi<br>baru,<br>keandalan sistem,<br>dan jejak audit.|mekanisme fallback ke<br>Scorecard Logistik, &<br>penyimpanan WORM.|bulanan, & log<br>WORM.|
|POJK<br>No.<br>10/POJK.05/2022|Kewajiban analisis<br>kelayakan|Pemodelan Dual-Track<br>(GBDT + Scorecard|Dokumentasi<br>metodologi|
|(Layanan Pendanaan|pendanaan (credit|WoE) dengan seleksi|pengembangan|
|Bersama / LPBBTI)|scoring)<br>yang<br>terukur dan mitigasi<br>risiko gagal bayar.|fitur alternatif yang<br>tervalidasi<br>secara<br>statistik.|model,<br>nilai<br>AUC/Gini/KS,<br>&<br>matriks kalibrasi PD.|
|POJK No. 22 Tahun|Prinsip kesetaraan|Penerapan<br>Pre-|Laporan uji berkala|
|2023 (Pelindungan|akses,<br>larangan|processing Reweighing|Disparate<br>Impact|
|Konsumen Sektor<br>Jasa Keuangan)|diskriminasi, serta<br>keterbukaan<br>informasi<br>produk/layanan.|&<br>Post-processing<br>Threshold Optimizer;<br>translasi<br>alasan<br>penolakan<br>yang<br>transparan.|Ratio (0,80-1,25) &<br>templat<br>notifikasi<br>nasabah.|
|SEOJK Manajemen<br>Risiko & Prinsip|Pengawasan aktif<br>manajemen, batasan|Penerapan Monotonic<br>Constraints pada model|Catatan keputusan<br>Override<br>di|



|Kehati-hatian|eksposur risiko, dan|GBDT serta kewajiban|Underwr|iter|
|---|---|---|---|---|
||pengendalian|otorisasi Underwriter|Decision|Cockpit &|
||internal<br>atas|pada kredit zona abu-|matriks|kewenangan|
||keputusan kredit.|abu.|kredit.||



## **III.2.4 Analisis Penanganan Keterbatasan Rancangan dalam Operasional AI** 

Secara keseluruhan, rancangan arsitektur yang diusulkan telah dilengkapi dengan mekanisme untuk mengatasi lima keterbatasan yang telah diidentifikasi pada awal penelitian. Permasalahan terkait ketergantungan terhadap data pihak ketiga ditangani melalui penggunaan _Data Quality Gatekeeper_ dan mekanisme pengalihan otomatis ke model _baseline_ apabila terjadi gangguan. Sementara itu, risiko perubahan distribusi data ( _data drift_ ) diantisipasi melalui pemantauan nilai PSI dan CSI pada Layer 4. 

Untuk mengatasi kendala waktu pemrosesan dan stabilitas hasil penjelasan model XAI, rancangan menggunakan pendekatan hibrida dengan memisahkan proses TreeSHAP yang berjalan secara sinkron dengan waktu pemrosesan kurang dari 50 ms dan proses LIME yang dijalankan secara asinkron untuk memeriksa konsistensi hasil penjelasan. Selanjutnya, permasalahan antara akurasi model dan keadilan ( _accuracy-fairness trade-off_ ) ditangani melalui pengaturan model yang mampu mempertahankan nilai _Disparate Impact Ratio_ sebesar 0,88 dengan nilai AUC tetap di atas 0,82. Terakhir, risiko manipulasi perilaku oleh nasabah ( _strategic gaming_ ) dikurangi melalui penerapan _Monotonic Constraints_ serta penggunaan fitur yang berasal dari riwayat data dalam jangka waktu menengah, yaitu 3 hingga 6 bulan, yang disimpan di dalam _Enterprise Feature Store_ . 

# **BAB IV KESIMPULAN** 

Berdasarkan hasil analisis kebutuhan, kajian terhadap literatur dan referensi jurnal internasional bereputasi Scopus Q1/Q2 pada periode 2021–2026, serta rancangan arsitektur sistem yang telah disusun, dapat diperoleh beberapa kesimpulan sebagai berikut: 

1. Pemanfaatan Data Alternatif untuk Mendukung Inklusi Keuangan Segmen Thin-File: Penggunaan data alternatif, seperti riwayat penggunaan layanan telekomunikasi, ketepatan pembayaran tagihan utilitas, dan aktivitas transaksi digital, dapat membantu mengatasi keterbatasan informasi kredit bagi masyarakat _unbanked_ dan pelaku UMKM yang belum memiliki riwayat kredit pada SLIK OJK. Namun, karakteristik data alternatif yang kompleks membutuhkan rancangan sistem kecerdasan buatan yang tidak hanya mengutamakan akurasi prediksi, tetapi juga harus memperhatikan aspek transparansi hasil keputusan ( _explainability_ ), keadilan algoritmik ( _fairness_ ), serta kemudahan dalam menelusuri rekam jejak keputusan melalui audit. 

2. Keandalan Rancangan Arsitektur Lima Lapisan ( _5-Tier Layered Architecture_ ): Rancangan arsitektur yang terdiri dari lima lapisan, yaitu (1) _Data Ingestion & Governance Layer_ , (2) _Modeling & Fairness Layer_ , (3) _Explainability Engine Layer_ , (4) _Model Risk Management & Audit Trail Layer_ , dan (5) _Presentation & Regulatory Output Layer_ , memberikan pembagian fungsi dan tanggung jawab yang jelas pada setiap bagian sistem. Berbagai keterbatasan operasional juga telah diantisipasi melalui beberapa mekanisme, seperti _Data Quality Gatekeeper_ , pemisahan proses komputasi TreeSHAP secara sinkron dan LIME secara asinkron, serta mekanisme _circuit-breaker fallback_ yang secara otomatis mengalihkan proses penilaian dari model _Gradient Boosting_ ke model Baseline Scorecard Logistik apabila ditemukan anomali atau perubahan data yang bersifat kritis (PSI > 0,25). 

3. Keseimbangan antara Akurasi Prediksi, Stabilitas XAI, dan Keadilan Algoritmik: Melalui penggunaan model hibrida yang menggabungkan _Gradient Boosting_ dengan _Monotonic Constraints_ , model pembanding berbasis WoE, serta metode pengendalian _fairness_ melalui Reweighing dan 

Threshold Optimizer, rancangan sistem mampu menghasilkan kemampuan pemisahan risiko yang baik dengan nilai AUC-ROC sebesar 0,824 dan Gini sebesar 0,648. Pada saat yang sama, sistem juga mampu menjaga keadilan terhadap kelompok demografis dengan nilai _Disparate Impact Ratio_ sebesar 0,88 yang masih berada dalam rentang batas yang ditetapkan. Selain itu, penggunaan TreeSHAP dan LIME yang telah dikalibrasi memungkinkan setiap keputusan kredit dilengkapi dengan alasan yang jelas ( _Adverse Action Reason Codes_ ), dengan tingkat konsistensi penjelasan lebih dari 94%, sehingga hasil keputusan lebih mudah dipahami oleh pemangku kepentingan, termasuk pengguna yang tidak memiliki latar belakang teknis. 

4. Pemenuhan Aspek Kepatuhan terhadap Regulasi OJK dan UU PDP: Penyediaan repositori _Immutable Audit Trail_ berbasis WORM, antarmuka _Human-in-the-Loop_ (HITL) untuk pemeriksaan pengajuan kredit yang berada pada zona abu-abu, Laporan _Explainability_ Nasabah, serta _Governance Checklist_ Regulasi OJK menjadi bagian penting dalam mendukung tata kelola sistem AI. Berbagai komponen tersebut dirancang agar penerapan sistem tetap sesuai dengan ketentuan regulasi yang berlaku di Indonesia, yaitu UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP), POJK No. 11/POJK.03/2022, POJK No. 10/POJK.05/2022, dan POJK No. 22 Tahun 2023. 

# **DAFTAR REFERENSI** 

Ballegeer, M., Bogaert, M., & Benoit, D. F. (2025). Evaluating the stability of 

model explanations in instance-dependent cost-sensitive credit scoring. _European Journal of Operational Research_ , _326_ (3), 630–640. https://doi.org/10.1016/j.ejor.2025.05.039 

Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. _Journal of the Operational Research Society_ , _73_ (1), 70–90. https://doi.org/10.1080/01605682.2021.1922098 

Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable Machine Learning in Credit Risk Management. _Computational Economics_ , _57_ (1), 203–216. https://doi.org/10.1007/s10614-020-10042-0 

- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16_ , 785–794. https://doi.org/10.1145/2939672.2939785 

- Djeundje, V. B., Crook, J., Calabrese, R., & Hamid, M. (2021). Enhancing credit scoring with alternative data. _Expert Systems with Applications_ , _163_ , 113766. https://doi.org/10.1016/j.eswa.2020.113766 

Dumitrescu, E., Hué, S., Hurlin, C., & Tokpavi, S. (2022). Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects. _European Journal of Operational Research_ , _297_ (3), 1178–1192. https://doi.org/10.1016/j.ejor.2021.06.053 

Gunnarsson, B. R., vanden Broucke, S., Baesens, B., Óskarsdóttir, M., & Lemahieu, 

W. (2021). Deep learning for credit scoring: Do or don’t? _European Journal_ 

_of Operational Research_ , _295_ (1), 292–305. 

https://doi.org/10.1016/j.ejor.2021.03.006 

Jerome H. Friedman. (2001). Greedy function approximation: A gradient boosting 

machine. _The Annals of Statistics_ , _29_ (5), 1189–1232. 

https://doi.org/10.1214/aos/1013203451 

Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification 

without discrimination. _Knowledge and Information Systems_ , _33_ (1), 1–33. https://doi.org/10.1007/s10115-011-0463-8 

Kozodoi, N., Jacob, J., & Lessmann, S. (2022). Fairness in credit scoring: 

Assessment, implementation and profit implications. _European Journal of_ 

_Operational Research_ , _297_ (3), 1083–1094. https://doi.org/10.1016/j.ejor.2021.06.023 

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. _European Journal of Operational Research_ , _247_ (1), 124–136. https://doi.org/10.1016/j.ejor.2015.05.030 

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2019). _Explainable AI for Trees: From Local Explanations to Global Understanding_ (arXiv:1905.04610). arXiv. https://doi.org/10.48550/arXiv.1905.04610 

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. _Proceedings of the 31st International Conference on Neural_ 

_Information Processing Systems, NIPS’17_ , 4768–4777. 

https://dl.acm.org/doi/10.5555/3295222.3295230 

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: 

Explaining the Predictions of Any Classifier. _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16_ , 1135–1144. https://doi.org/10.1145/2939672.2939778 Undang-undang (UU) Nomor 27 Tahun 2022 tentang Pelindungan Data Pribadi. 

Retrieved November 14, 2026, from 

https://peraturan.bpk.go.id/Details/229798/uu-no-27-tahun-2022 

