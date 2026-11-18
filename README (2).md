# Wo hört Whisper hin?
### Zeitliche und spektrale Lokalisierung der Evidenz für einzelne Wörter in einem Encoder-Decoder-ASR-Modell

**Typ:** Post-hoc-Analyse · **Modelle:** `openai/whisper-tiny`, `openai/whisper-base` · **Daten:** LibriSpeech `test-clean` mit Wortgrenzen aus dem Montreal Forced Aligner (MFA) · **Code:** [`whisper_xai.ipynb`](whisper_xai.ipynb)

## 1 Fragestellung

> Welche Zeitabschnitte und akustischen Merkmale sind für die Transkription einzelner Wörter durch Whisper entscheidend, und wie gut lokalisieren Input-Attribution und Attention diese Evidenz im Vergleich zu den tatsächlichen Wortgrenzen?

Die meisten XAI-Auswertungen haben keine objektive Referenz. Hier gibt es eine: Forced Alignment liefert für jedes Wort ein Zeitsegment $[s, e)$. Damit lässt sich messen, ob eine Erklärung „an der richtigen Stelle“ liegt.

| | Hypothese | Prüfung |
|---|---|---|
| **H1** | Die Attribution für ein Wort liegt deutlich über Zufallsniveau im Wortsegment, plus wenige 100 ms Kontext. | `mass_in` vs. `chance` (Wilcoxon), Pointing Game, ereigniskorreliertes Profil |
| **H2** | Nur wenige Cross-Attention-Köpfe des Decoders lokalisieren gut, und es sind die von OpenAI für Zeitstempel genutzten `alignment_heads`. | Pointing-Accuracy pro Kopf |
| **H3** | Frühe Encoder-Schichten arbeiten lokal und diagonal, späte global. | Lokalität und Aufmerksamkeitsdistanz pro Schicht (Spearman) |
| **H4** | Attention ist schwächer kausal mit der Ausgabe verknüpft als Attribution. | Deletion-Kurven, Kopf-Ablation |

## 2 Modell und Erklärungsziel

Whisper (Radford et al., 2023) wandelt Audio in ein 80-Band-Log-Mel-Spektrogramm um: 10 ms pro Frame, **immer auf 30 s aufgefüllt**, also 3000 Frames. Zwei Conv-Schichten halbieren die Zeitauflösung, der Encoder arbeitet dann mit 1500 Frames à 20 ms. Ein autoregressiver Decoder liest den Encoder über Cross-Attention.

* **Ziel der Erklärung:** $f(X)=\log p(\text{Wort}\mid X)=\sum_{t\in\text{Wort}}\log p(y_t\mid y_{<t},X)$. Gerechnet wird mit Teacher Forcing auf Whispers eigene Transkription. Ausgewertet werden nur Wörter, die Whisper korrekt erkannt hat, denn nur dort ist die MFA-Grenze die richtige Referenz. BPE-Tokens werden zu Wörtern zusammengefasst.
* **Baseline $X_0$ = Stille:** das Padding-Spektrum der jeweiligen Äußerung. Der Feature-Extractor setzt Werte unter (Maximum − 8) auf diesen Mindestwert, aufgefüllte Nullen landen also genau dort. $X_0$ ist damit für das Modell bekannte Stille und keine künstliche Null-Eingabe.

## 3 Methoden

| Methode | Was sie berechnet | Warum | Schwäche |
|---|---|---|---|
| **Integrated Gradients** (Sundararajan et al., 2017) | $(x-x_0)\odot\int_0^1\nabla f(x_0+\alpha(x-x_0))\,d\alpha$, 32 Stützstellen | Erfüllt die Axiome *Completeness* und *Sensitivity*. Die Completeness wird im Notebook pro Wort geprüft. | Hängt von der Baseline ab, teuer |
| **Gradient × Δ** (Shrikumar et al., 2016) | $\nabla f(x)\odot(x-x_0)$ | Billige lokale Näherung, Vergleich zu IG | Wird bei Sättigung ungenau |
| **Zeit-Occlusion** (Zeiler & Fergus, 2014) | Abfall von $\log p$, wenn ein 100-ms-Fenster durch $X_0$ ersetzt wird | Modellagnostisch und kausal | Grobe Auflösung, erzeugt Eingaben außerhalb der Trainingsverteilung |
| **Band-Occlusion** | Wie oben, aber 8 Mel-Bandgruppen nur im Wortsegment | Kausale Gegenprobe zur spektralen IG-Analyse | Wie oben |
| **Cross-Attention** | Attention der Decoder-Queries eines Wortes über die Encoder-Frames, pro Schicht und Kopf | Whisper erzeugt seine Wort-Zeitstempel aus genau diesem Signal (alignment heads + DTW) | Zeigt Informationsfluss, keine Wichtigkeit (Jain & Wallace, 2019) |
| **Analyse der Encoder-Köpfe** (Clark et al., 2019; Voita et al., 2019) | Pro Kopf: Lokalität (±40 ms), Vorgänger/Nachfolger, mittlere Distanz, Entropie, Masse auf dem Padding, Bindung an das eigene Wort (`within_lift`), Masse auf Pausen | Beschreibt, welche Art von Lokalisierung im Encoder stattfindet | Nur beschreibend |
| **Kopf-Ablation** (Michel et al., 2019) | Ausgabe eines Kopfes vor `out_proj` auf 0 setzen → Δ log p, ΔWER | Kausaler Test für H2 und H4 | Null-Ablation ist selbst ein Eingriff außerhalb der Verteilung |

**Metriken:** Ein Relevanzprofil $p(t)$ wird auf Summe 1 normiert, für Attribution als |·| über alle Mel-Bänder summiert. Daraus berechnen wir:

* `mass_in`: Masse im Wortsegment,
* `lift`: `mass_in` geteilt durch den Zufallswert (Wortdauer / Äußerungsdauer),
* Pointing Game: Liegt das Maximum im Wort (±20 ms)?,
* Masse in den 300 ms vor bzw. nach dem Wort,
* Abstand des Peaks zur Wortmitte.

Das **ereigniskorrelierte Profil** mittelt die Relevanz über alle Wörter auf einer relativen Zeitachse (0 = Wortanfang, 1 = Wortende). Der **Deletion-Test** (Samek et al., 2017) entfernt die laut Methode relevantesten Frames, und zwar 0,25- bis 4-mal so viele wie das Wort lang ist. Als Schranken dienen *Zufall* und *Oracle*, wobei Oracle die Frames nach Nähe zur Wortmitte entfernt.

## 4 Ausführung

```
Windows lokal (GTX 1660 SUPER): `python -m pip install -r requirements.txt`
```
Die erste Notebook-Zelle installiert PyTorch mit CUDA 12.8. Danach den Kernel neu starten und das Notebook von oben ausführen. Die GPU-Zelle bricht ab, wenn CUDA nicht verfügbar ist, statt unbemerkt auf der CPU weiterzurechnen.
* `QUICK = True`: Probelauf mit 10 Äußerungen.
* Standard: 100 Äußerungen ≤ 15 s für die Lokalisierung, 25 davon für Deletion und Ablation.
* Alle Parameter stehen in `CFG`.
* Ausgaben landen in `results/`: Abbildungen, `*_word_metrics.csv`, `model_comparison.csv` und `summary.json`.

## 5 Ergebnisse

> *Nach dem Lauf aus `results/model_comparison.csv` und den Abbildungen eintragen. Die Leitfragen stehen in Notebook-Abschnitt 14.*

| | tiny | base |
|---|---|---|
| WER | | |
| IG `lift` (Median) / Pointing (Zufall) | | |
| Bester Cross-Kopf / Mittel der alignment heads (Pointing) | | |
| Encoder `local`: erste → letzte Schicht | | |
| Deletion-AUC: IG / CrossAttn / Oracle / Zufall | | |
| ΔWER: alignment heads vs. Zufallsköpfe | | |

**H1–H4:** jeweils bestätigt, teilweise bestätigt oder widerlegt, mit Begründung über die Kennzahl.

## 6 Grenzen und Reflexion

* **Was als Referenz gilt:** MFA-Grenzen sind selbst modellbasiert und auf etwa 10–20 ms genau. Ein Profil, das über die Grenze hinausreicht, ist nicht automatisch falsch, denn wegen Koartikulation steckt akustische Information über ein Wort auch in den Nachbarsegmenten.
* **Abhängigkeit von der Baseline:** IG und Occlusion beantworten die Frage „Relevanz *relativ zu Stille*“. Eine andere Baseline, etwa Rauschen, kann andere Karten liefern.
* **Out-of-Distribution-Problem** (Hooker et al., 2019): Das Entfernen von Frames erzeugt Eingaben, die das Modell im Training nie gesehen hat. Ein Abfall von $\log p$ kann daher teilweise auf diese Verteilungsverschiebung zurückgehen und nicht auf fehlende Information. Die Stille-Baseline mildert das, beseitigt es aber nicht.
* **Selektionsbias:** Weil nur korrekt erkannte Wörter ausgewertet werden, fehlen die schwierigen Fälle. Funktionswörter sind zudem stark vom Kontext des Decoder-Sprachmodells bestimmt. Eine schlechte akustische Lokalisierung ist dort plausibel und kein Fehler der Methode.
* **Attention ist nicht gleich Erklärung:** Attention auf dem 30-s-Padding (*Attention Sink*) oder hohe Lokalisierung ohne Wirkung bei der Ablation zeigen genau diese Grenze. H4 macht sie messbar.
* **Skalierung:** Auch `whisper-base` ist klein. Ob die Befunde auf `large` übertragbar sind, bleibt offen.

## Quellen

* Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML. arXiv:2212.04356
* Panayotov, V. et al. (2015). *LibriSpeech: An ASR corpus based on public domain audio books.* ICASSP.
* McAuliffe, M. et al. (2017). *Montreal Forced Aligner.* Interspeech. Alignments: Lugosch, L. et al. (2019), Interspeech; HF-Datensatz `gilkeyio/librispeech-alignments`
* Sundararajan, M., Taly, A., Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML. arXiv:1703.01365
* Shrikumar, A. et al. (2016). *Not Just a Black Box: Learning Important Features Through Propagating Activation Differences.* arXiv:1605.01713
* Zeiler, M., Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.* ECCV. arXiv:1311.2901
* Samek, W. et al. (2017). *Evaluating the Visualization of What a Deep Neural Network Has Learned.* IEEE TNNLS. arXiv:1509.06321
* Hooker, S. et al. (2019). *A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR).* NeurIPS. arXiv:1806.10758
* Becker, S. et al. (2018/2024). *Interpreting and Explaining Deep Neural Networks for Classification of Audio Signals (AudioMNIST).* arXiv:1807.03418
* Clark, K. et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP. arXiv:1906.04341
* Voita, E. et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL. arXiv:1905.09418
* Michel, P., Levy, O., Neubig, G. (2019). *Are Sixteen Heads Really Better than One?* NeurIPS. arXiv:1905.10650
* Jain, S., Wallace, B. (2019). *Attention is not Explanation.* NAACL. arXiv:1902.10186
* Wiegreffe, S., Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP. arXiv:1908.04626
