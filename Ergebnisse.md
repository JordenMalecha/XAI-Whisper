# Ergebnisse: Wo hört Whisper hin?

Auswertung des Notebooks [`whisper_xai.ipynb`](whisper_xai.ipynb) für `openai/whisper-tiny` und `openai/whisper-base`. Grundlage sind die in `results/` gespeicherten Dateien (`summary.json`, `model_comparison.csv`, `whisper-*_word_metrics.csv`, Abbildungen). Methoden, Hypothesen und Quellen stehen in [`README (2).md`](README%20(2).md).

**Herkunft der Zahlen**

* **Exakt:** Werte aus `summary.json`, `model_comparison.csv` und eigene Auswertungen der Wort-CSV-Dateien (Wilcoxon-Tests, Wortarten, Wortlänge, IG-Fehler).
* **Aus Abbildungen abgelesen („ca.“):** Deletion-Kurven, Ablations-Heatmaps, Spektral- und ERP-Plots, Encoder-Heatmaps. Dafür wurden keine Rohdaten gespeichert.
* **Nicht gespeichert:** Tabellen der Kopf-Ablation (Gruppenablation im Detail), Spearman-Tests über die Schichten, Median-Spearman |IG| gegen Energie, Mann-Whitney-Tests aus Abschnitt 8 des Notebooks. Sie stehen nur in den Notebook-Ausgaben und werden hier nicht angegeben, soweit sie nicht neu berechnet wurden.

## 1 Versuchsaufbau

| Parameter | Wert |
|---|---|
| Daten | LibriSpeech `test-clean` (`gilkeyio/librispeech-alignments`), Referenz: MFA-Wortgrenzen |
| Äußerungen | 100 (≤ 15 s), Seed 0 |
| Teilmenge für Deletion und Ablation (`n_faith`) | 25 |
| Integrated Gradients | 32 Stützstellen, Batch 8 |
| Zeit-Occlusion | Fenster 100 ms, Schritt 50 ms |
| Band-Occlusion | 8 Mel-Bandgruppen, Wortsegment ±50 ms |
| Pointing-Toleranz `tol` | 20 ms |
| Kontextfenster `ctx` | 300 ms |
| Deletion-Anteile | 0,25 / 0,5 / 1 / 2 / 4 × Wortlänge |
| Kopf-Ablation mit WER (Generierung) | ja |
| Hardware | CPU (Laufzeit `whisper-tiny`, Hauptschleife: 10 h 16 min) |

**Stichprobe**

| | whisper-tiny | whisper-base |
|---|---|---|
| Äußerungen | 100 | 100 |
| Korrekt erkannte, analysierte Wörter | 1751 | 1774 |
| davon Funktionswörter / Inhaltswörter | 912 / 839 | 919 / 855 |
| WER (auf normalisierten Wörtern) | 0,084 | 0,066 |
| Median Wortdauer | 0,25 s | 0,25 s |
| Zufallsniveau `chance` (Wortdauer / Äußerungsdauer) | 0,043 | 0,043 |
| Zufallsniveau Pointing (`hit_chance`) | 0,049 | 0,049 |

Ausgewertet werden nur korrekt erkannte Wörter (Selektionsbias, siehe Abschnitt 11).

## 2 Qualität der Integrated Gradients (Completeness)

Die Completeness-Prüfung vergleicht $\sum_i \mathrm{IG}_i$ mit $f(x)-f(x_0)$. Das Notebook nennt einen relativen Fehler unter ca. 5 % als Ziel.

| | whisper-tiny | whisper-base |
|---|---|---|
| Relativer Fehler, Median | 0,317 | 0,279 |
| Relativer Fehler, Mittelwert | 0,451 | 0,410 |
| Relativer Fehler, 90. Perzentil | 0,933 | 0,899 |
| Anteil der Wörter mit Fehler < 5 % | 11,5 % | 13,2 % |

**Befund:** Die IG-Näherung mit 32 Stützstellen ist nicht konvergiert. Der mediane Fehler liegt bei 28–32 % und damit weit über dem Zielwert. Alle IG-Ergebnisse (Abschnitte 4–9) gelten daher mit Vorbehalt. Für belastbare IG-Aussagen wäre `ig_steps` deutlich zu erhöhen (z. B. 128 oder mehr) und der Fehler erneut zu prüfen. Die Schlüsse zu Occlusion, Cross-Attention und Ablation sind davon nicht betroffen.

## 3 Gesamtübersicht: Modellvergleich

Quelle: `results/model_comparison.csv`.

| Kennzahl | whisper-tiny | whisper-base |
|---|---|---|
| WER | 0,084 | 0,066 |
| IG `mass_in` (Mittel) | 0,232 | 0,208 |
| IG `lift` (Median) | 6,309 | 5,433 |
| IG Pointing-Accuracy | 0,751 | 0,736 |
| Pointing Zufall | 0,049 | 0,049 |
| Bester Cross-Attention-Kopf (Pointing) | 0,991 | 0,991 |
| Mittel aller Cross-Attention-Köpfe (Pointing) | 0,312 | 0,220 |
| Mittel der `alignment_heads` (Pointing) | 0,579 | 0,730 |
| Encoder `local`, Mittel Schicht 0 | 0,408 | 0,350 |
| Encoder `local`, Mittel letzte Schicht | 0,162 | 0,115 |
| Encoder `pad_mass` (Mittel aller Köpfe) | 0,075 | 0,120 |
| Deletion-AUC IG (mittl. Δ log p) | −3,650 | −2,807 |
| Deletion-AUC Oracle | −4,303 | −3,819 |
| ΔWER bei Ablation der `alignment_heads` | 1,383 | 0,868 |
| ΔWER bei Ablation gleich großer Zufallsmengen (Mittel über 5) | 0,538 | 2,770 |

## 4 TF1 – Zeitliche Lokalisierung (Input-Attribution und Attention)

### 4.1 Kennzahlen pro Methode

Quelle: `results/summary.json` (Mittelwerte; `lift` und Peak-Offset als Median). `n` = Anzahl Wörter.

**whisper-tiny** (n = 1751)

| Methode | `mass_in` | `lift` (Median) | Pointing | Masse davor | Masse danach | Peak-Offset |
|---|---|---|---|---|---|---|
| Occlusion | 0,641 | 18,6 | 0,915 | 0,095 | 0,092 | −20 ms |
| CrossAttn (Alignment-Köpfe) | 0,472 | 12,5 | 0,678 | 0,050 | 0,272 | +60 ms |
| IG | 0,232 | 6,3 | 0,751 | 0,105 | 0,102 | −15 ms |
| Grad×Δ | 0,212 | 5,3 | 0,740 | 0,091 | 0,107 | +5 ms |
| CrossAttn (alle Köpfe) | 0,249 | 7,3 | 0,031 | 0,147 | 0,109 | −1115 ms |

**whisper-base** (n = 1774)

| Methode | `mass_in` | `lift` (Median) | Pointing | Masse davor | Masse danach | Peak-Offset |
|---|---|---|---|---|---|---|
| Occlusion | 0,566 | 15,7 | 0,864 | 0,111 | 0,096 | −25 ms |
| CrossAttn (Alignment-Köpfe) | 0,563 | 17,0 | 0,852 | 0,134 | 0,105 | −20 ms |
| IG | 0,208 | 5,4 | 0,736 | 0,100 | 0,097 | −20 ms |
| Grad×Δ | 0,178 | 4,4 | 0,633 | 0,087 | 0,091 | −5 ms |
| CrossAttn (alle Köpfe) | 0,187 | 5,4 | 0,050 | 0,123 | 0,092 | −908 ms |

Zufallsniveau: `mass_in` = 0,043, `lift` = 1, Pointing = 0,049.

### 4.2 Statistische Tests (Wilcoxon, einseitig `mass_in` > `chance`)

Für alle Methoden und beide Modelle gilt $p < 10^{-287}$ (n ≈ 1750 pro Test). Die Hypothese H1 (Lokalisierung über Zufallsniveau) wird damit für jede Methode klar bestätigt. Der Wert $10^{-288}$ ist ein numerischer Grenzwert und sagt nur, dass praktisch alle Wörter über Zufall liegen. Zur Effektstärke sind `lift` und Pointing aussagekräftiger.

### 4.3 Vor- und Nachkontext (gepaarter Wilcoxon-Test, Mittel der Differenz „danach − davor“)

| Modell | Methode | Δ (danach − davor) | p |
|---|---|---|---|
| tiny | IG | −0,003 | 0,0017 |
| tiny | Grad×Δ | +0,016 | 4,4·10⁻¹⁴ |
| tiny | Occlusion | −0,004 | 0,171 (n. s.) |
| tiny | CrossAttn (Alignment-Köpfe) | **+0,222** | 9,5·10⁻²⁵⁶ |
| tiny | CrossAttn (alle Köpfe) | −0,037 | 5,6·10⁻⁹⁹ |
| base | IG | −0,003 | 0,0010 |
| base | Grad×Δ | +0,003 | 0,151 (n. s.) |
| base | Occlusion | −0,015 | 1,8·10⁻⁶ |
| base | CrossAttn (Alignment-Köpfe) | −0,029 | 4,5·10⁻⁴⁰ |
| base | CrossAttn (alle Köpfe) | −0,031 | 2,3·10⁻¹¹⁹ |

* Bei IG, Grad×Δ und Occlusion ist der Unterschied zwischen Vor- und Nachkontext praktisch null (|Δ| ≤ 0,016). Ein signifikanter Test bei n ≈ 1750 bedeutet hier keine relevante Asymmetrie. Die Attribution ist symmetrisch um das Wort verteilt, ohne Überhang in den Nachkontext.
* **Auffällig:** Die Alignment-Köpfe von `whisper-tiny` legen 27,2 % der Masse in die 300 ms **nach** dem Wort, aber nur 5,0 % davor, bei einem Peak-Offset von +60 ms. In `whisper-base` ist dieser Versatz nicht vorhanden (−20 ms, 13,4 % davor, 10,5 % danach).

### 4.4 Ereigniskorreliertes Profil (`whisper-tiny`)

Abbildung: [results/erp.png](results/erp.png). Die Relevanz ist relativ zur Gleichverteilung normiert (1 = Zufall). Werte ca. aus der Abbildung.

| Methode | Peak (relative Zeit) | Peak-Höhe | Beobachtung |
|---|---|---|---|
| Occlusion | ca. 0,55 | ca. 29,5 | Schmalster und höchster Peak, schneller Abfall nach dem Wortende |
| CrossAttn (Alignment-Köpfe) | ca. 0,6 | ca. 21 | Langer Ausläufer nach dem Wort (ca. 7–8 bei u = 1,2–1,5, noch ca. 3 bei u = 3) |
| CrossAttn (alle Köpfe) | ca. 0,5 | ca. 10 | Breit, schon vor dem Wort ansteigend |
| IG | ca. 0,5 | ca. 9 | Breiter Peak im Wort, flacher Ausläufer |
| Grad×Δ | ca. 0,5 | ca. 7,7 | Ähnlich IG, etwas niedriger |

Alle Profile liegen deutlich im Wortsegment (0–1). Ein „Vokalkern-Peak“ in der Wortmitte ist bei allen Methoden zu sehen. Auffällig stark ist die Wirkung in den ersten Wortanteilen, bei Occlusion steigt die Kurve schon ab ca. u = −0,2. Alle Profile fallen nach dem Wortende auf ein Niveau von etwa 2–4 und nicht auf 1 ab. Das passt zu Koartikulation und zum bidirektionalen Encoder.

**Funktions- gegenüber Inhaltswörtern (IG)**

* Funktionswörter haben ein höheres und breiteres Profil (Peak ca. 10,5 gegen ca. 7,8 bei Inhaltswörtern) und auch außerhalb des Wortes mehr Relevanz (ca. 2,5–4 vor und nach dem Wort, gegen ca. 1–2).
* Funktionswörter sind kürzer. Weil das Profil auf die Gleichverteilung normiert ist, erhalten kurze Wörter bei gleicher absoluter Masse höhere Werte. Der Befund zeigt also eher mehr Kontextabhängigkeit der kurzen Wörter als bessere Lokalisierung.

### 4.5 Wortarten, Wortlänge und Tokenzahl

Eigene Auswertung der `*_word_metrics.csv`. Funktionswörter sind kurz, reduziert und stark kontextabhängig.

**Pointing-Accuracy (Trefferquote) nach Wortart**

| Modell | Methode | Funktionswörter | Inhaltswörter |
|---|---|---|---|
| tiny | IG | 0,713 | 0,793 |
| tiny | Grad×Δ | 0,635 | 0,855 |
| tiny | Occlusion | 0,901 | 0,931 |
| tiny | CrossAttn (Alignment-Köpfe) | 0,433 | 0,945 |
| base | IG | 0,676 | 0,801 |
| base | Grad×Δ | 0,522 | 0,752 |
| base | Occlusion | 0,856 | 0,871 |
| base | CrossAttn (Alignment-Köpfe) | 0,955 | 0,742 |

**`mass_in` (Mittel) und `lift` (Median) nach Wortart**

| Modell | Methode | `mass_in` Funktion | `mass_in` Inhalt | `lift` Funktion | `lift` Inhalt |
|---|---|---|---|---|---|
| tiny | IG | 0,156 | 0,313 | 7,13 | 5,62 |
| tiny | Occlusion | 0,577 | 0,711 | 28,5 | 12,9 |
| tiny | CrossAttn (Alignment-Köpfe) | 0,264 | 0,699 | 11,96 | 12,90 |
| base | IG | 0,136 | 0,285 | 5,94 | 4,94 |
| base | Occlusion | 0,505 | 0,632 | 24,9 | 11,1 |
| base | CrossAttn (Alignment-Köpfe) | 0,496 | 0,635 | 25,1 | 11,4 |

* Funktionswörter haben bei IG und Occlusion eine geringere absolute Masse im Wort (`mass_in` ca. halb so groß bei IG), aber einen höheren `lift`. Der Grund ist das Zufallsniveau: Kurze Wörter haben ein kleineres `chance`. Mann-Whitney (IG, `lift` Funktion gegen Inhalt): tiny $p = 3{,}6\cdot10^{-18}$, base $p = 4{,}0\cdot10^{-14}$.
* **Besonderheit `whisper-tiny`:** Bei Funktionswörtern treffen die Alignment-Köpfe nur zu 43,3 % (Inhaltswörter: 94,5 %), mit starker Verschiebung nach dem Wort. In `whisper-base` ist es umgekehrt (95,5 % gegen 74,2 %).
* Die Wortdauer korreliert stark mit `mass_in` (IG, Spearman): tiny ρ = 0,70 ($p < 10^{-259}$), base ρ = 0,72 ($p < 10^{-285}$). Der Zusammenhang ist teilweise mechanisch, weil `chance` mit der Dauer wächst, deshalb gibt der `lift` die bessere Aussage.

**Pointing-Accuracy (IG) nach Anzahl BPE-Tokens**

| Tokens | tiny: Trefferquote (n) | base: Trefferquote (n) |
|---|---|---|
| 1 | 0,739 (1650) | 0,722 (1661) |
| 2 | 0,931 (87) | 0,935 (92) |
| 3 | 1,000 (12) | 1,000 (19) |
| 4 | 1,000 (2) | 1,000 (2) |

Mehrtoken-Wörter werden besser lokalisiert. Sie sind länger, das Zufallsniveau ist dadurch aber ebenfalls höher. Die Gruppen mit 3 oder 4 Tokens sind für Aussagen zu klein.

Abbildung: [results/tf1_wordtypes.png](results/tf1_wordtypes.png).

### 4.6 Beispiel-Äußerung

Abbildung: [results/example.png](results/example.png) (`whisper-tiny`, Satz „This is the explanation of the shallows which are found in that part of the Atlantic Ocean.“, Fokuswort *explanation*).

* Die Cross-Attention der Alignment-Köpfe läuft als saubere Diagonale durch die Äußerung und folgt den MFA-Wortgrenzen eng. Die Prefix-Tokens (`<|en|>`, `<|transcribe|>`, `<|notimestamps|>`) und `<|endoftext|>` ziehen Attention auf den Anfang und das Ende des Padding-Bereichs.
* Die IG-Karte für *explanation* liegt überwiegend im Wortsegment, hat aber auch Reste in den beiden Wörtern davor („this is the“). Das Profil ist deutlich unruhiger und diffuser als das der Cross-Attention.
* Im Zeitprofil haben Cross-Attention und IG ihre Maxima im Wort, bei den Spitzen sind sie nicht deckungsgleich.

## 5 TF2 – Spektrale Merkmale (`whisper-tiny`)

Abbildung: [results/tf2_spectral.png](results/tf2_spectral.png). Werte ca. aus der Abbildung. Die Frequenzen sind Mittenfrequenzen der Mel-Bins.

### 5.1 IG-Anteil gegen Energieanteil

* Das Verhältnis |IG|-Anteil / Energieanteil liegt **über 1** bei ca. 200–500 Hz (bis ca. 1,3, Grundfrequenz und erste Harmonische), bei ca. 1,2–3,5 kHz (ca. 1,2–1,5, zweiter und dritter Formant) und an der obersten Bandkante um 7,7 kHz (ca. 2,3, vermutlich ein Randeffekt des letzten Mel-Bins).
* Es liegt **unter 1** unter ca. 150 Hz (0,4–0,7) und bei ca. 800–1000 Hz (0,6–0,7). Dazwischen und im Bereich 4–7 kHz liegt es bei ca. 0,85–1.
* Die Kurven für Funktions- und Inhaltswörter sind fast deckungsgleich. IG zeichnet also nicht nur die Energie nach. Es hebt Bereiche hervor, die für Sprache typisch sind (Grundfrequenz, Formantbereich), und dämpft den Bereich mit niedrigster Energie.

### 5.2 Band-Occlusion (Abfall von log p im Wortsegment)

| Band (Hz) | Inhaltswort | Funktionswort |
|---|---|---|
| 37–372 | ca. 0,41 | ca. 0,23 |
| **410–745** | **ca. 1,12** | **ca. 0,62** |
| 782–1128 | ca. 0,55 | ca. 0,17 |
| 1173–1657 | ca. 0,60 | ca. 0,27 |
| 1722–2433 | ca. 0,46 | ca. 0,19 |
| 2528–3571 | ca. 0,40 | ca. 0,22 |
| 3711–5244 | ca. 0,07 | ca. 0,02 |
| 5449–7699 | ca. 0,12 | ca. 0,07 |

* Das **wichtigste Band ist 410–745 Hz** (erster Formant), bei Inhalts- und Funktionswörtern. Ein Entfernen verringert $\log p$ am stärksten.
* Alle Bänder bis ca. 3,6 kHz tragen Evidenz. Bänder über 3,7 kHz haben fast keinen Einfluss.
* Inhaltswörter reagieren in jedem Band stärker als Funktionswörter, in den mittleren Bändern etwa doppelt bis dreifach. Funktionswörter sind weniger von einem einzelnen Band abhängig.
* Der Befund passt zu den IG-Ergebnissen: Die Bereiche mit IG-Anteil über Energieanteil (200–500 Hz, 1,2–3,5 kHz) fallen mit den Bändern zusammen, deren Entfernen $\log p$ senkt.

## 6 TF3 – Cross-Attention-Köpfe des Decoders

Abbildung: [results/tf3_cross_heads.png](results/tf3_cross_heads.png). Werte der Pointing-Accuracy pro Kopf (Zufall ≈ 0,05), ★ = von OpenAI definierter `alignment_head`. Werte aus der Abbildung.

### 6.1 whisper-tiny (4 Schichten × 6 Köpfe)

| Schicht | Kopf 0 | Kopf 1 | Kopf 2 | Kopf 3 | Kopf 4 | Kopf 5 |
|---|---|---|---|---|---|---|
| 0 | 0,00 | 0,07 | 0,01 | 0,00 | 0,11 | 0,00 |
| 1 | 0,03 | 0,00 | 0,08 | 0,38 | 0,05 | 0,01 |
| 2 | 0,46 | 0,11 | 0,59 ★ | 0,71 | 0,02 | 0,99 |
| 3 | 0,76 ★ | 0,99 | 0,30 ★ | 0,71 ★ | 0,50 ★ | 0,61 ★ |

### 6.2 whisper-base (6 Schichten × 8 Köpfe)

| Schicht | K0 | K1 | K2 | K3 | K4 | K5 | K6 | K7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0,00 | 0,03 | 0,00 | 0,00 | 0,00 | 0,06 | 0,01 | 0,01 |
| 1 | 0,01 | 0,02 | 0,08 | 0,02 | 0,02 | 0,05 | 0,01 | 0,00 |
| 2 | 0,12 | 0,04 | 0,08 | 0,02 | 0,03 | 0,12 | 0,13 | 0,00 |
| 3 | 0,06 | 0,99 ★ | 0,03 | 0,33 | 0,17 | 0,00 | 0,20 | 0,00 |
| 4 | 0,31 | 0,40 | 0,85 ★ | 0,65 ★ | 0,02 | 0,18 | 0,01 | 0,85 ★ |
| 5 | 0,42 | 0,97 ★ | 0,04 ★ | 0,62 | 0,71 ★ | 0,49 | 0,77 ★ | 0,61 |

### 6.3 Befunde

* **Wenige Köpfe lokalisieren gut, und zwar nur in den letzten Schichten.** Mit Pointing > 0,5 sind es in `whisper-tiny` 7 von 24 Köpfen (alle in Schicht 2–3) und in `whisper-base` 9 von 48 (alle in Schicht 3–5). In den frühen Schichten liegt das Pointing bei Zufall oder knapp darüber. Der beste Kopf erreicht in beiden Modellen 0,99.
* **Alignment-Köpfe:** Mittleres Pointing 0,579 (tiny, 6 Köpfe) und 0,730 (base, 8 Köpfe), im Mittel deutlich über dem Durchschnitt aller Köpfe (0,312 und 0,220). Die Hypothese H2 wird damit grundsätzlich gestützt.
* **Aber nicht exakt:**
  * In `whisper-tiny` sind zwei der drei besten Köpfe keine Alignment-Köpfe: Schicht 3 / Kopf 1 (0,99) und Schicht 2 / Kopf 5 (0,99). Außerdem liegt Schicht 2 / Kopf 3 mit 0,71 gleichauf mit dem besten Alignment-Kopf in Schicht 3. Der schwächste Alignment-Kopf (Schicht 3 / Kopf 2) erreicht nur 0,30.
  * In `whisper-base` hat der Alignment-Kopf Schicht 5 / Kopf 2 ein Pointing von nur 0,04 und gleichzeitig 0,88 Masse auf dem Padding. Der Kopf wirkt wie ein Attention Sink.
* **Padding-Masse:** In den frühen Decoder-Schichten liegt ein großer Teil der Cross-Attention auf dem 30-s-Padding (Schicht 0: bis 0,87 in tiny und 0,88 in base). Das ist das typische Sink-Muster. Die Alignment-Köpfe von `whisper-tiny` haben dagegen wenig Padding-Masse (0,05–0,29).
* **Mittelung über alle Köpfe:** Der Peak liegt im Mittel ca. 0,9–1,1 s neben dem Wort (Pointing 0,03 und 0,05, also Zufall), weil sich Sink-Köpfe und frühe Schichten mit den lokalisierenden Köpfen mischen. Mit `lift` 7,3 und 5,4 und `mass_in` 0,25 und 0,19 bleibt trotzdem etwas Masse im Wort. Das Mittel aller Köpfe ist als Lokalisierer nicht brauchbar.

## 7 TF4 – Encoder-Self-Attention

Abbildungen: [results/tf4_enc_whisper-tiny.png](results/tf4_enc_whisper-tiny.png), [results/tf4_enc_whisper-base.png](results/tf4_enc_whisper-base.png), [results/tf4_enc_maps.png](results/tf4_enc_maps.png) (Beispiel-Attention-Karten, `whisper-tiny`).

### 7.1 Kennzahlen

| | whisper-tiny | whisper-base |
|---|---|---|
| `local`, Mittel Schicht 0 | 0,408 | 0,350 |
| `local`, Mittel letzte Schicht | 0,162 | 0,115 |
| `pad_mass`, Mittel aller Köpfe | 0,075 | 0,120 |

### 7.2 Befunde

* **H3 wird in der Richtung bestätigt:** Die lokale Attention (±40 ms) nimmt von der ersten zur letzten Schicht ab (tiny: 0,41 → 0,16, base: 0,35 → 0,12). Die letzten Schichten arbeiten globaler.
* **Wortgebundene lokale Köpfe in Schicht 0:** In `whisper-tiny` haben Schicht 0 / Kopf 2 und Kopf 5 `local` ≈ 0,95 und `within_lift` ≈ 15 (Spitzenwerte der Karte). In `whisper-base` gilt das für Schicht 0 / Kopf 3 und Kopf 4 (`local` ≈ 0,95, `within_lift` ≈ 15). Die Beispiel-Karten zeigen für diese Köpfe eine reine Diagonale (Nachfolger- und Vorgänger-Muster bei Schicht 0 / Kopf 2 und Kopf 5 in `whisper-tiny`).
* **Mittlere Attention-Distanz (`mean_dist`):** Einzelne Köpfe in späteren Schichten erreichen ca. 2 s. Die Karten zeigen aber auch Köpfe, die selbst in Schicht 3 noch fast diagonal arbeiten (z. B. Schicht 3 / Kopf 1 und Kopf 4 und 5 in `whisper-tiny`). Der Übergang lokal → global gilt also im Mittel, nicht für jeden einzelnen Kopf.
* **Attention Sinks:** Die Masse auf dem Padding ist im Mittel klein. In `whisper-base` fällt Schicht 5 / Kopf 1 mit `pad_mass` ≈ 0,55 und `sil_lift` ≈ 3 auf. Er legt viel Masse auf Padding und Pausen. In `whisper-tiny` liegt das Maximum bei ca. 0,2.
* **Heuristische Kopf-Typisierung:** Das Notebook hat die Köpfe im Beispiel mit der Heuristik aus Abschnitt 11 klassifiziert. In `whisper-tiny` ergibt sie nur wenige Treffer für „Vorgänger“, „Nachfolger“ und „lokal“ (Schicht 0 / Kopf 2 und Kopf 5, Schicht 2 / Kopf 1 und Schicht 3 / Kopf 1). Fast alle anderen Köpfe landen in „global“, auch solche mit klarer Diagonale (z. B. Schicht 2 / Kopf 0 und Kopf 3). Die Schwellen sind damit für die gemessene Attention zu streng. Als Beschreibung der Köpfe ist besser die Kennzahlen-Karte zu lesen.

## 8 TF5 – Faithfulness

### 8.1 Deletion-Test

Abbildung: [results/tf5_deletion.png](results/tf5_deletion.png). Gemessen wird Δ log p(Wort), wenn k × Wortlänge viele der laut Methode relevantesten Frames auf Stille gesetzt werden. Niedriger (negativer) = die Methode findet die wichtigen Frames besser.

**AUC (mittleres Δ log p über alle Anteile)** aus `model_comparison.csv`:

| | whisper-tiny | whisper-base |
|---|---|---|
| IG | −3,650 | −2,807 |
| Oracle | −4,303 | −3,819 |

**Kurven ca. aus der Abbildung (Δ log p)**

whisper-tiny

| Methode | 0,25 × | 1 × | 2 × | 4 × |
|---|---|---|---|---|
| Oracle | ca. −0,3 | ca. −6,0 | ca. −7,0 | ca. −6,5 |
| Occlusion | ca. −0,9 | ca. −4,5 | ca. −5,7 | ca. −6,0 |
| IG | ca. −0,5 | ca. −4,0 | ca. −5,7 | ca. −6,3 |
| CrossAttn (Alignment-Köpfe) | ca. −0,3 | ca. −3,9 | ca. −5,7 | ca. −6,1 |
| Grad×Δ | ca. −0,4 | ca. −3,6 | ca. −5,6 | ca. −6,3 |
| CrossAttn (alle Köpfe) | ca. −0,1 | ca. −3,0 | ca. −6,0 | ca. −6,4 |
| Zufall | ca. 0 | ca. −0,1 | ca. −0,4 | ca. −1,4 |

whisper-base

| Methode | 0,25 × | 1 × | 2 × | 4 × |
|---|---|---|---|---|
| Oracle | ca. −0,2 | ca. −5,0 | ca. −6,6 | ca. −6,2 |
| Occlusion | ca. −0,6 | ca. −3,1 | ca. −4,6 | ca. −5,0 |
| IG | ca. −0,2 | ca. −2,8 | ca. −4,5 | ca. −5,6 |
| CrossAttn (Alignment-Köpfe) | ca. −0,2 | ca. −2,8 | ca. −5,5 | ca. −6,0 |
| Grad×Δ | ca. −0,2 | ca. −2,2 | ca. −4,2 | ca. −5,4 |
| CrossAttn (alle Köpfe) | ca. 0 | ca. −1,4 | ca. −4,1 | ca. −5,7 |
| Zufall | ca. 0 | ca. 0 | ca. −0,1 | ca. −0,8 |

**Befunde**

* **Alle Methoden schlagen Zufall deutlich.** Auch bei 4 × Wortlänge liegt der Zufall nur bei ca. −1,4 (tiny) und ca. −0,8 (base), während alle Methoden ca. −5 bis −6,5 erreichen.
* **Keine Methode erreicht das Oracle.** Beim Anteil 1 × Wortlänge liegt das Oracle bei ca. −6,0 (tiny) und ca. −5,0 (base), die besten Methoden bei ca. −4,5 und ca. −3,1. Das Oracle ist die MFA-Wortmitte, also die „Wahrheit“ laut Alignment. Frames in der Wortmitte zu entfernen schadet mehr, als es jede Methode vorhersagt.
* **Occlusion ist am treuesten** bei kleinen Anteilen (ca. 0,25–1 ×). Das ist zu erwarten, weil Occlusion genau diese Größe misst.
* **H4a (Teil 1) wird nicht bestätigt:** Die Cross-Attention der **Alignment-Köpfe** ist in der Deletion so treu wie IG. Bei `whisper-tiny` liegt sie praktisch gleichauf, bei `whisper-base` ist sie bei 2 × Wortlänge sogar steiler als IG und Occlusion (ca. −5,5 gegen ca. −4,5 und −4,6). Nur der **Mittelwert über alle Köpfe** ist deutlich schwächer (ca. −3,0 gegen ca. −4,0 bei 1 × in tiny, und ca. −1,4 gegen ca. −2,8 in base).
* Grad×Δ liegt bei kleinen Anteilen knapp unter IG und ist etwas weniger treu.
* `whisper-base` ist bei gleichen Anteilen robuster gegen das Entfernen von Frames als `whisper-tiny` (IG-AUC −2,81 gegen −3,65).

### 8.2 Kopf-Ablation

Abbildungen: [results/tf5_ablation_whisper-tiny.png](results/tf5_ablation_whisper-tiny.png), [results/tf5_ablation_whisper-base.png](results/tf5_ablation_whisper-base.png). Gemessen wird die Änderung des mittleren Token-log p bei Teacher Forcing, wenn ein einzelner Kopf auf 0 gesetzt wird (25 Äußerungen). Werte aus den Heatmaps.

**Encoder-Self-Attention**

| Modell | Stärkste Effekte (Δ log p) | Übrige Köpfe |
|---|---|---|
| tiny | Schicht 0: Kopf 4 = −5,57, Kopf 2 = −4,99, Kopf 3 = −2,79, Kopf 5 = −2,20 | alle anderen zwischen −0,66 und 0 |
| base | Schicht 0 / Kopf 1 = −0,24, Schicht 0 / Kopf 6 = −0,14 | alle anderen zwischen −0,08 und 0 |

**Decoder-Cross-Attention**

| Modell | Stärkste Effekte (Δ log p) | Übrige Köpfe |
|---|---|---|
| tiny | Schicht 1 / Kopf 2 = −2,90, Schicht 0 / Kopf 4 = −1,20 | alle anderen ≥ −0,14 |
| base | Schicht 2 / Kopf 2 = −1,13, Schicht 3 / Kopf 0 = −0,23 | alle anderen ≥ −0,08 |

**Lokalisierung gegen kausale Wirkung (Cross-Attention-Köpfe, Spearman ρ)**

| Modell | ρ | p |
|---|---|---|
| tiny | −0,56 | 0,0046 |
| base | −0,54 | 7,4·10⁻⁵ |

**Befunde**

* **Die kausal wichtigsten Köpfe sind nicht die, die lokalisieren.** Die stärksten Einbrüche der Cross-Attention treten bei Köpfen mit Pointing ≈ 0,08–0,11 auf (tiny: Schicht 1 / Kopf 2 und Schicht 0 / Kopf 4, base: Schicht 2 / Kopf 2). Alle Köpfe mit hohem Pointing (> 0,5) bewirken bei der Ablation dagegen nur Änderungen von ca. 0 bis −0,15. Das stützt **H4b**: Lokalisierungsgüte sagt die kausale Wirkung nicht voraus.
* Das negative ρ (−0,56 und −0,54) ist als Rangkorrelation zu lesen. Viele Köpfe ohne Lokalisierung haben eine Wirkung von praktisch genau 0, während gut lokalisierende Köpfe kleine, aber von 0 verschiedene Effekte haben. Das ρ ist dadurch negativ, obwohl die großen Effekte bei schlecht lokalisierenden Köpfen liegen. Das ist meine Deutung des Streudiagramms, die Rohdaten sind nicht gespeichert.
* **Encoder:** Die Wirkung ist in Schicht 0 konzentriert und bei `whisper-tiny` sehr groß. Die Ablation von Schicht 0 / Kopf 4 ändert das Token-log p um −5,57. Das sind gerade die sehr lokalen, diagonalen Köpfe (Abschnitt 7). In `whisper-base` ist die Wirkung einzelner Encoder-Köpfe dagegen klein (maximal −0,24), das Modell ist redundanter.

### 8.3 Gruppenablation: Alignment-Köpfe gegen Zufallsköpfe (ΔWER)

Quelle: `model_comparison.csv`. ΔWER ist die Änderung der WER nach der Ablation gegenüber der unveränderten Hypothese (25 Äußerungen).

| | whisper-tiny | whisper-base |
|---|---|---|
| Alignment-Köpfe gemeinsam abgeschaltet | +1,383 | +0,868 |
| Zufallsköpfe gleicher Zahl (Mittel über 5 Mengen) | +0,538 | +2,770 |

* **whisper-tiny:** Das Abschalten der Alignment-Köpfe schadet rund 2,6-mal so stark wie gleich große Zufallsmengen. Die Alignment-Köpfe sind hier kausal besonders wichtig.
* **whisper-base:** Das Bild ist umgekehrt, Zufallsmengen schaden im Mittel mehr als die Alignment-Köpfe (+2,77 gegen +0,87). Eine Erklärung, die ich aus den Daten nur vermute: Unter den fünf Zufallsmengen enthalten einzelne den sehr wichtigen Kopf Schicht 2 / Kopf 2 (Δ log p = −1,13, Abschnitt 8.2), was den Mittelwert stark hebt. Die Einzelwerte der fünf Mengen sind nicht gespeichert, so dass sich das nicht prüfen lässt.
* ΔWER-Werte über 1 bedeuten, dass sich der Wortfehler mehr als verdoppelt hat. Bei freier Generierung können Wiederholungen oder Halluzinationen die WER stark erhöhen. Mit nur 25 Äußerungen sind die Werte auch noch recht ungenau.
* Insgesamt lässt sich nicht sagen, dass die Alignment-Köpfe über beide Modelle hinweg kausal wichtiger sind als zufällige Köpfe.

## 9 Skalierung: tiny gegen base

| Aspekt | Ergebnis |
|---|---|
| Erkennung | `whisper-base` ist besser (WER 0,066 gegen 0,084). |
| Input-Attribution | Beide Modelle liegen bei IG ähnlich (Pointing 0,75 und 0,74). `lift` und `mass_in` sind bei base etwas niedriger (5,4 gegen 6,3 und 0,208 gegen 0,232), die Attribution ist also etwas diffuser. Occlusion ist bei base auch etwas schwächer (`lift` 15,7 gegen 18,6). |
| Alignment-Köpfe | Deutlich besser in `whisper-base` (Pointing 0,730 gegen 0,579). Ihre Cross-Attention lokalisiert dort mit Trefferquote 0,852 gegen 0,678 und zeigt keinen Nachlauf-Versatz. |
| Anteil guter Köpfe | In `whisper-base` sind relativ weniger Köpfe lokalisierend (Mittel 0,220 gegen 0,312, 9 von 48 gegen 7 von 24). Der beste Kopf erreicht in beiden Modellen 0,99. |
| Encoder-Lokalität | In beiden Modellen sinkt `local` mit der Tiefe. `whisper-base` ist in jeder Schicht etwas weniger lokal. |
| Attention Sinks | `pad_mass` ist bei base höher (0,120 gegen 0,075). |
| Deletion | `whisper-base` ist robuster (IG-AUC −2,81 gegen −3,65; Oracle −3,82 gegen −4,30). |
| Ablation einzelner Köpfe | In `whisper-tiny` hängt das Modell stark von einzelnen Encoder-Köpfen der Schicht 0 ab (bis −5,6), in `whisper-base` kaum (bis −0,24). Das größere Modell ist redundanter. |
| Alignment-Köpfe, Gruppenablation | tiny: Alignment-Köpfe wichtiger als Zufall. base: umgekehrt (siehe 8.3). |

Beide Modelle sind klein. Aussagen über größere Whisper-Modelle lassen sich daraus nicht ableiten.

## 10 Beantwortung der Hypothesen

| | Hypothese | Ergebnis | Begründung |
|---|---|---|---|
| **H1** | Die Attribution liegt über Zufall im Wortsegment, mit etwas Kontext. | **Bestätigt** (mit Vorbehalt für IG) | Alle Methoden: $p < 10^{-287}$ für `mass_in` > `chance`. `lift` 4,4–18,6, Pointing 0,63–0,92 (ohne Mittelung über alle Köpfe). Der Kontext vor und nach dem Wort ist bei IG, Grad×Δ und Occlusion etwa gleich groß (je ca. 0,09–0,11). Die IG-Completeness ist aber nicht erreicht (Median-Fehler ca. 30 %). Occlusion lokalisiert am schärfsten (`mass_in` 0,57–0,64 gegen 0,21–0,23 bei IG). |
| **H2** | Nur wenige Cross-Attention-Köpfe lokalisieren, und es sind die Alignment-Köpfe. | **Teilweise bestätigt** | Wenige Köpfe (7/24 und 9/48 mit Pointing > 0,5), nur in den letzten Schichten, Bestwert 0,99. Alignment-Köpfe lokalisieren im Mittel deutlich besser als der Durchschnitt (0,58 und 0,73 gegen 0,31 und 0,22). Aber in `whisper-tiny` liegen zwei der drei besten Köpfe außerhalb der Alignment-Köpfe, und in `whisper-base` ist ein Alignment-Kopf ein Padding-Sink (Pointing 0,04). |
| **H3** | Frühe Encoder-Schichten sind lokal, späte global. | **Im Mittel bestätigt** | `local` sinkt von 0,41 auf 0,16 (tiny) und von 0,35 auf 0,12 (base). Die Schicht 0 enthält sehr lokale, wortgebundene Köpfe (`local` ≈ 0,95, `within_lift` ≈ 15). Einzelne Köpfe in späten Schichten bleiben lokal. Die Spearman-Tests über die Schichten sind nicht gespeichert. |
| **H4** | Attention ist schwächer kausal mit der Ausgabe verknüpft als Attribution. | **Gemischt** | (a) Deletion: nicht bestätigt für die Alignment-Köpfe (gleich treu wie IG, bei base teils treuer). Bestätigt nur für den Mittelwert über alle Köpfe. (b) Ablation: bestätigt, denn die kausal wichtigsten Köpfe lokalisieren schlecht, und gut lokalisierende Köpfe haben kaum Wirkung. |

## 11 Grenzen und Fehlerquellen

* **IG nicht konvergiert:** Median-Fehler der Completeness ca. 28–32 %, nur 11–13 % der Wörter unter 5 % (Abschnitt 2). IG-Zahlen sind daher unsicher. Bei Occlusion, Cross-Attention und Ablation gibt es dieses Problem nicht.
* **Out-of-Distribution-Effekt:** Das Entfernen von Frames (Occlusion, Deletion) und das Nullsetzen von Köpfen erzeugen Eingaben, die Whisper nie gesehen hat. Ein Abfall von $\log p$ kann teilweise daran liegen und nicht an fehlender Information. Die Stille-Baseline mildert das nur.
* **Baseline-Abhängigkeit:** IG und Occlusion messen Relevanz relativ zu Stille. Andere Baselines können andere Karten liefern.
* **Referenzgenauigkeit:** MFA-Wortgrenzen sind selbst modellbasiert (ca. 10–20 ms), die Toleranz für das Pointing Game liegt bei 20 ms. Wegen Koartikulation steckt akustische Information über ein Wort auch außerhalb seiner Grenzen, ein Ausläufer ist nicht automatisch ein Fehler.
* **Selektionsbias:** Nur korrekt erkannte Wörter werden ausgewertet. Falsch erkannte Wörter, bei denen Lokalisierung besonders interessant wäre, fehlen. Funktionswörter sind kurz und stark vom Sprachmodell-Kontext des Decoders bestimmt.
* **Zufallsniveau und Wortlänge:** `mass_in` hängt von der Wortdauer ab (Spearman ca. 0,7). Für Vergleiche zwischen Wortarten sind `lift` und Pointing besser geeignet, auch sie sind aber vom Zufallsniveau abhängig.
* **Stichprobengröße:** 100 Äußerungen für Lokalisierung, aber nur 25 für Deletion und Ablation. Die Ablationswerte (insbesondere ΔWER) sind entsprechend ungenau. Es gibt keine Konfidenzintervalle oder Wiederholungen mit anderen Seeds.
* **Teacher Forcing:** Die Attributionen erklären die Wahrscheinlichkeit der eigenen Hypothese unter Teacher Forcing, nicht den freien Generierungsprozess.
* **Attention ist keine Erklärung** (Jain & Wallace, 2019): Hohe Lokalisierung ohne kausale Wirkung (Abschnitt 8.2) bestätigt diese Grenze in den Daten.
* **Kopf-Typisierung:** Die Schwellen der Heuristik sind für die gemessene Attention zu streng (Abschnitt 7.2).
* **Plot-Werte:** Zahlen mit „ca.“ sind aus Abbildungen abgelesen und auf etwa ±0,1–0,2 genau.
* **Tiny-Modell:** Die Plots TF1 bis TF2 und das ereigniskorrelierte Profil beziehen sich nur auf das erste Modell (`MAIN = whisper-tiny`). Für `whisper-base` liegen dazu keine Plots vor.

## 12 Fazit

1. Whisper hört im Wesentlichen **im Wortsegment** hin. Alle Attributionsmethoden und die gut lokalisierenden Cross-Attention-Köpfe liegen weit über Zufall. Ein symmetrischer Kontext von wenigen 100 ms ist erkennbar (Koartikulation).
2. **Occlusion lokalisiert am schärfsten** und ist bei kleinen Löschanteilen am treuesten. IG ist diffuser und hier nicht konvergiert, Grad×Δ ist noch etwas schwächer.
3. Spektral zählt vor allem der **Bereich um 410–745 Hz** (erster Formant), gefolgt vom Bereich bis ca. 3,6 kHz. Über 3,7 kHz trägt kaum Evidenz. Inhaltswörter reagieren in allen Bändern stärker als Funktionswörter.
4. **Zeitstempel-nahe Cross-Attention gibt es nur in wenigen Köpfen der letzten Decoder-Schichten.** Die `alignment_heads` lokalisieren im Mittel gut, sind aber nicht exakt die besten Köpfe, und der Durchschnitt über alle Köpfe ist als Lokalisierer unbrauchbar.
5. Der **Encoder** arbeitet in Schicht 0 mit sehr lokalen, wortgebundenen Köpfen und wird zu den späteren Schichten hin globaler.
6. **Lokalisierung ist nicht Wichtigkeit.** Köpfe mit hohem Pointing haben bei der Ablation kaum Wirkung, die kausal wichtigsten Köpfe lokalisieren schlecht.
7. **Skalierung:** `whisper-base` ist genauer, hat besser lokalisierende Alignment-Köpfe, ist robuster gegen Deletion und gegen die Ablation einzelner Köpfe, hat aber diffusere Attribution und mehr Padding-Sinks.

**Offene Punkte für eine Wiederholung:** `ig_steps` erhöhen und Completeness prüfen, mehr als 25 Äußerungen für Deletion und Ablation, Rohdaten der Ablation und Deletion speichern, Plots für TF1 und TF2 auch für `whisper-base`, Gruppenablation mit Einzelwerten der Zufallsmengen.
