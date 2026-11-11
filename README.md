# Immortal Time: behandelt, weil man lange genug lebte – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-immortal-time-demo.streamlit.app/)**

Teil der Linie **Überlebenszeit** (Zeit bis zum Ereignis bei Beobachtungen, die irgendwann abbrechen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Der Aufhänger ist ein Container mit einer **Lebenszeit** (Tage bis zum Ausfall seines Kühlaggregats; ausgedachte Welt) und einer **Behandlung** (hier: eine Wartung), die irgendwann nach der Ankunft beginnt, sobald ein Techniker Zeit hat.
Wer die Container einteilt in „je behandelt“ und „nie behandelt“ und beide **ab Tag 0** vergleicht, rechnet die Wartezeit als behandelt: Ein Container musste den Beginn **erleben**, um in die erste Gruppe zu kommen.
Diese **unsterbliche Zeit** (*immortal time*) lässt die Behandlung schützend aussehen, auch wenn sie nichts bewirkt. Die Demo führt das an sechs Containern von Hand vor, misst die Scheinwirkung in einer simulierten Welt
(und wie sie von der Wartezeit abhängt), vergleicht **Landmarken** (Status zu einem festen Tag) und die **zeitabhängige Auswertung** (aufgeteilte Person-Zeit) mit der Wahrheit der Welt.

Gleicher Aufhänger Containerverweilzeit wie [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo), dort ohne Behandlung und ohne Zensierung. Die Kante in der Linie: von der Cox-Regression
([cox-regression-demo](https://github.com/sebastian-hanisch/cox-regression-demo)) zu einem Merkmal, dessen Wert sich im Lauf der Zeit ändert. Die Zahlen sind ausgedachte Zahlen; es werden keine Zahlen aus echten Terminals behauptet.

**Annahme in dieser Demo, sichtbar in der App und hier: Die Zensierung ist unabhängig vom Ereignis.** In dieser Welt gibt es keine Zensierung (jeder Container wird bis zum Ausfall beobachtet); die Annahme wird hier nicht gebraucht und nicht geprüft.

## Warum dieses Problem

Wer wissen will, ob eine Behandlung schützt, die erst nach einer Wartezeit beginnt, teilt die Container oft nach „wurde irgendwann behandelt“. In der Welt dieser Demo (Lebenszeit ohne Behandlung lognormal mit Median 4 Tagen und σ = 0,9,
Behandlungsbeginn exponentialverteilt mit Mittel 4 Tagen, **keine Wirkung der Behandlung** (β = 0), 1000 Container, 200 Läufe, Zufalls-Seed 35) findet die naive Cox-Auswertung β̂ = −1,278 ± 0,006 (Hazard-Verhältnis 0,28):
Die Behandlung scheint die Ausfallrate auf das 0,28-fache zu senken, obwohl sie nichts verändert. 62 % der Container zählen als behandelt. Die Landmarke Tag 2 findet β̂ = 0,005 ± 0,005, die Landmarke Tag 4 −0,010 ± 0,007,
die zeitabhängige Auswertung −0,006 ± 0,005. Die Kernfrage ist: Wie viel Schutz findet man, wenn man Container danach einteilt, ob sie je behandelt wurden, und was hilft? Das Hazard-Verhältnis 0,28 gilt nur für diese Welt
(keine Wirkung, Wartezeit 4 Tage, Lebenszeit Median 4 Tage, n = 1000), nicht allgemein.

## Befunde und Korrekturen gegenüber der ersten Messung

Vormessung = eine erste, schnelle Messung vor dem Bau der Demo, mit dem Zufallsgenerator von NumPy und 200 Läufen. Die Zahlen daraus stehen hier nur als Vergleichswerte; es gilt die Zahl dieser Demo.

1. **Zahlen weichen ab, es gilt die Zahl dieser Demo.** Vormessung (numpy-Zufall): naiv −1,276 ± 0,005 bzw. −1,274 (zwei Messzellen), Landmarke Tag 2 −0,002 bzw. 0,002, zeitabhängig −0,008; mit Wirkung naiv −1,719,
   Landmarke Tag 2 / 4 −0,192 / −0,189 (802 / 570 Container), zeitabhängig −0,512. Diese Demo (SplitMix64, Zufalls-Seed 35, 200 Läufe, gleiche Welt) steht in der Tabelle unten; die Abstände zur Vormessung liegen im
   Bereich der Streuung zwischen Zufalls-Seeds (Stabilitätstabelle unten).
2. **Die ursprüngliche Erwartung war „die naive Verzerrung wächst mit der Wartezeit“; die Messung zeigt das Gegenteil.** Bei Wahrheit β = 0 liegt das naive β̂ bei einer mittleren Wartezeit von 0,5 Tagen bei −2,75 und fällt über 1, 2, 4, 8 auf −0,93
   bei 16 Tagen (Tabelle unten, 100 Läufe je Wartezeit); es verschwindet in keiner der sechs Wartezeiten. Die Scheinwirkung ist **bei der kürzesten gemessenen Wartezeit (0,5 Tage) am größten**. Bei noch kürzerer Wartezeit sind fast alle Container behandelt; ein Test mit 0,1 Tagen (15 Läufe, ohne Wirkung, naive Auswertung) meldet dann in einem Teil der Läufe Trennung statt einer Zahl, also kein Ergebnis (nur dieser Test, keine Messzelle). Unsere Lesart (Deutung, nicht gemessen):
   Bei kurzer Wartezeit zählen fast alle Container als behandelt, und die wenigen unbehandelten sind genau die, die vor ihrem Beginn ausfallen. Die ursprüngliche Erwartung „ohne Wartezeit verschwindet die Verzerrung“
   gilt nur für eine zufällige, von der Lebenszeit unabhängige **Vorab-Zuteilung**, bei der der Status am Tag 0 feststeht (Messung unten: −0,001 ± 0,004), nicht für eine Wartezeit, die im Erzeuger gegen 0 geht.
3. **Die Größe des Verlusts der Landmarke hängt in dieser Welt (β = −0,5, Wartezeit 4 Tage) kaum am τ, aber an der Wartezeit.** Die erste Messung nannte −0,19 statt −0,50 (Wartezeit 4 Tage). In der Reihe über die Wartezeit (β = −0,5) findet die Landmarke Tag 2 bei 0,5 Tagen −0,05,
   bei 16 Tagen −0,33; über alle 16 Landmarken von τ = 0,5 bis 8 Tage (Wartezeit 4 Tage) liegt das mittlere β̂ nur zwischen −0,201 und −0,166, während die Streuung zwischen den Läufen von 0,09 auf 0,21 wächst
   und die Zahl der verbleibenden Container von 990 auf 324 fällt.
4. **Neu: die Deutung „Verdünnung“ ist geprüft.** Zählt man die bei τ noch Unbehandelten, die später behandelt werden, nicht als unbehandelt weiter, sondern zensiert sie bei ihrem Beginn, findet die Landmarke die Wirkung:
   Tag 2 −0,501 ± 0,007, Tag 4 −0,509 ± 0,009 (Wahrheit −0,5, 200 Läufe). Der Verlust der gewöhnlichen Landmarke kommt in dieser Welt also aus diesem Weiterzählen (Eigenmessung).
5. **Neu: Überdeckung und Instabilität.** Das 95-%-Intervall der Landmarke enthält die Wahrheit bei β = −0,5 in 0 % (Tag 2) und 10 % (Tag 4) der Läufe, das der zeitabhängigen Auswertung in 96 % (Eigenmessung; die Landmarke schätzt
   streng genommen eine andere Größe als β, siehe Ehrliche Grenzen). Bei Wartezeit 0,5 Tage hat die Landmarke Tag 4 in 83 von 100 Läufen kein Ergebnis (perfekte Trennung: fast alle Container sind bis dahin behandelt), die Zahl in der Tabelle
   mittelt dann nur 17 Läufe.
6. **Neu: schädliche Behandlung.** Bei Wahrheit β = +0,5 findet die naive Auswertung −0,866 ± 0,006 (scheinbarer Schutz), Landmarke Tag 2 0,283 ± 0,005, Tag 4 0,261 ± 0,007, zeitabhängig 0,496 ± 0,005.
7. **Handbeispiel anders gelesen.** Die ursprüngliche Erwartung war, die Behandlung wirke „hier nirgends“. In der App steht nur, was die Rechnung zeigt (Landmarke 3,5: 7,0 gegen 7,0); sechs ausgedachte Container sagen nichts über eine Welt.
8. **Nicht gebaut:** Ein Vergleich mit einem fremden Paket (lifelines, statsmodels sind im Test-Umfeld nicht installiert); stattdessen die Orakel unten, unabhängig geschriebene Verfahren.

## Modell

Container $i$ hat die Lebenszeit $T_i$ und den Behandlungsbeginn $S_i$ (die Behandlung wirkt nur, wenn $S_i < T_i$). Die Rate ist $h_0(t)$ bis $S_i$ und $h_0(t)\,e^{\beta}$ danach; $h_0$ gehört zur Lognormalverteilung (Median 4 Tage,
σ = 0,9), $S_i$ ist exponentialverteilt (Mittel einstellbar, Voreinstellung 4 Tage), unabhängig von der Lebenszeit ohne Behandlung (bei β ≠ 0 hängt die beobachtete Lebenszeit über die Wirkung von $S_i$ ab). Der Erzeuger zieht $E_i \sim \mathrm{Exp}(1)$ und setzt $H_0(T_i) = E_i$ für $E_i < H_0(S_i)$, sonst
$H_0(T_i) = H_0(S_i) + (E_i - H_0(S_i))\,e^{-\beta}$; die Umkehrung von $H_0$ braucht die Umkehrfunktion der Normalverteilung (selbst geschrieben, gegen scipy nur in den Tests geprüft). Drei Auswertungen, jeweils eine Cox-Regression (Breslow)
mit einem 0/1-Merkmal „behandelt“ und alle Ausfälle beobachtet:

- **naiv:** eine Zeile $(0, T_i]$ je Container, Status $\mathbf 1[S_i < T_i]$ für die ganze Zeit;
- **Landmarke τ:** nur Container mit $T_i > \tau$, Zeile $(\tau, T_i]$, Status $\mathbf 1[S_i \le \tau]$; spätere Behandlungen bleiben unbeachtet;
- **zeitabhängig:** Zeile $(0, S_i]$ unbehandelt ohne Ausfall und Zeile $(S_i, T_i]$ behandelt mit Ausfall (wenn $S_i < T_i$), sonst $(0, T_i]$ unbehandelt mit Ausfall.

Formeln, Lesehilfe und Handbeispiel stehen im Expander **📐 Mathematische Formulierung** der App; „Wo die Annahmen enden“ und „Annahmen, Grenzen und bekannte Probleme“ ebenfalls.

## Methodik: die vier Experimente der App

1. **Von Hand.** Sechs ausgedachte Container mit (Lebenszeit, Behandlungsbeginn) = (2, 1), (3, 5), (4, 2,5), (6, –), (8, 7), (10, 3). Naiv sind die Container 1, 3, 5, 6 behandelt: Mittel der Lebenszeit **6,0** gegen **4,5** der unbehandelten
   (Container 2, 4). Mit Landmarke τ = 3,5 bleiben die Container 3, 4, 5, 6; behandelt (Beginn bis 3,5) sind 3 und 6: **7,0** gegen **7,0**. Der Regler wählt τ; die Tabellen stehen in Brüchen (`tests/test_model.py`, exakt).
2. **Die naive Einteilung und die Wartezeit** (Kernfrage). 200 Läufe mit festen Zufalls-Seeds; Seitenleiste: wahre Wirkung β (−1,0 bis +0,5), mittlere Wartezeit (0,5 bis 16 Tage), Container n (100 bis 2000), Zufalls-Seed.
   Darunter die Reihe über die mittlere Wartezeit (0,5 bis 16 Tage, 100 Läufe je Wartezeit, gleiche Zufallszahlen, nur die Wartezeit ändert sich) für naiv, Landmarke Tag 2 und 4 und zeitabhängig.
3. **Landmarke wählen.** Regler τ (0,5 bis 8 Tage); verbleibende Container, davon behandelte, β̂ (Mittel über die Läufe), Anteil der wahren Wirkung, den die Landmarke findet; dazu die Reihe über alle 16 Landmarken.
4. **Zeitabhängig.** Die Person-Zeit der sechs Handcontainer wird in zehn Zeilen aufgeteilt (Tabelle in der App); alle drei Auswertungen auf denselben Datensätzen im Vergleich, mit Überdeckung des 95-%-Intervalls.

Die Zielstudien-Emulation (Klonen, Zensieren, Gewichten) wird in der App nur als Suchtreffer genannt und nicht gerechnet.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tools/measure_immortal_time_result.json` (erzeugt mit `tools/measure_immortal_time.py`; Läufe: Lauf $r$ hat den Zufalls-Seed `run_seed(Zufalls-Seed, r)` aus SplitMix64) und werden in `tests/test_claims.py` neu gerechnet.
Welt, wenn nichts anderes steht: Lebenszeit lognormal (Median 4 Tage, σ = 0,9), Behandlungsbeginn Exp (Mittel 4 Tage), 1000 Container, Zufalls-Seed 35, 200 Läufe, keine Zensierung. β̂ als Mittel über die Läufe ± Standardfehler des Mittels;
„Überdeckung“: Anteil der Läufe, in denen β̂ ± 1,96·Standardfehler (Modell) den wahren Wert β der Welt enthält.

**Hauptergebnis** (je Zeile: Wahrheit β = 0 und Wahrheit β = −0,5; „bleiben“ = mittlere Zahl verbleibender Container der Landmarke, erst bei β = 0, dann bei β = −0,5):

| Auswertung | Wahrheit β = 0: β̂ (Mittel ± Standardfehler) | Überdeckung | Wahrheit β = −0,5: β̂ | Überdeckung |
|---|---|---|---|---|
| naiv (je behandelt, ab Tag 0) | −1,278 ± 0,006 | 0 % | −1,720 ± 0,006 | 0 % |
| Landmarke Tag 2 (bleiben: 779 / 801) | 0,005 ± 0,005 | 97 % | −0,190 ± 0,005 | 0 % |
| Landmarke Tag 4 (bleiben: 500 / 567) | −0,010 ± 0,007 | 95 % | −0,201 ± 0,007 | 10 % |
| zeitabhängig (aufgeteilte Person-Zeit) | −0,006 ± 0,005 | 97 % | −0,507 ± 0,005 | 96 % |

Die Zahlen der verbleibenden Container (779 / 801 bei Tag 2, 500 / 567 bei Tag 4) weichen leicht von denen der Wartezeit-Reihen bei 4 Tagen unten ab (778 / 800 und 501 / 568). Vermutliche Ursache (Deutung, nicht einzeln geprüft): Diese Tabelle mittelt 200 Läufe, die Reihen 100 Läufe je Wartezeit.

- Naiv: β̂ = −1,278 ± 0,006 (Hazard-Verhältnis 0,28) ohne Wirkung und −1,720 ± 0,006 (Hazard-Verhältnis 0,18) mit Wirkung β = −0,5; das Intervall enthält die Wahrheit in keinem der 200 Läufe.
- Landmarke ohne Wirkung: Tag 2 0,005 ± 0,005, Tag 4 −0,010 ± 0,007 (jeweils innerhalb von 2 Standardfehlern des Mittels von 0). Mit Wirkung findet sie nur einen Teil: 38 % bzw. 40 % der Wirkung (β̂ geteilt durch β, Verhältnis der Mittel; Eigenmessung).
- Zeitabhängig: ohne Wirkung −0,006 ± 0,005, mit Wirkung −0,507 ± 0,005 (Wahrheit −0,5).

**Stabilität über sechs Zufalls-Seeds** (je 200 Läufe, Mittel von β̂; Standardwelt):

| Zufalls-Seed | naiv β̂ (β = 0) | Landmarke Tag 2 (β = 0) | zeitabhängig (β = 0) | naiv β̂ (β = −0,5) | Landmarke Tag 2 (β = −0,5) | zeitabhängig (β = −0,5) |
|---|---|---|---|---|---|---|
| 35 | −1,278 | 0,005 | −0,006 | −1,720 | −0,190 | −0,507 |
| 101 | −1,275 | 0,006 | 0,003 | −1,714 | −0,186 | −0,497 |
| 202 | −1,269 | 0,000 | −0,002 | −1,709 | −0,194 | −0,501 |
| 303 | −1,279 | −0,003 | −0,011 | −1,720 | −0,197 | −0,512 |
| 404 | −1,273 | −0,001 | −0,003 | −1,713 | −0,192 | −0,504 |
| 505 | −1,275 | 0,003 | 0,008 | −1,714 | −0,189 | −0,491 |

Über die sechs Zufalls-Seeds liegt das naive β̂ bei β = 0 zwischen −1,279 und −1,269, die Landmarke Tag 2 bei β = −0,5 zwischen −0,197 und −0,186 und die zeitabhängige Auswertung bei β = −0,5 zwischen −0,512 und −0,491.

**Reihe über die mittlere Wartezeit, Wahrheit β = 0** (100 Läufe je Wartezeit, 1000 Container, Zufalls-Seed 35; „behandelt“: Anteil der Container mit Behandlungsbeginn vor dem Ausfall):

| Mittlere Wartezeit (Tage) | behandelt (naiv) | naiv β̂ | Landmarke Tag 2: β̂ (bleiben) | Landmarke Tag 4: β̂ (bleiben) | zeitabhängig β̂ |
|---|---|---|---|---|---|
| 0,5 | 97 % | −2,75 | −0,01 (778) | −0,53 (501), 83 Läufe ohne Ergebnis | −0,01 |
| 1,0 | 91 % | −2,09 | 0,01 (778) | −0,03 (501) | −0,01 |
| 2,0 | 79 % | −1,59 | 0,00 (778) | 0,00 (501) | −0,02 |
| 4,0 | 62 % | −1,28 | 0,01 (778) | −0,01 (501) | −0,01 |
| 8,0 | 43 % | −1,06 | 0,01 (778) | 0,00 (501) | 0,00 |
| 16,0 | 27 % | −0,93 | 0,00 (778) | 0,00 (501) | −0,01 |

**Reihe über die mittlere Wartezeit, Wahrheit β = −0,5** (gleiche Voraussetzungen):

| Mittlere Wartezeit (Tage) | behandelt (naiv) | naiv β̂ | Landmarke Tag 2: β̂ (bleiben) | Landmarke Tag 4: β̂ (bleiben) | zeitabhängig β̂ |
|---|---|---|---|---|---|
| 0,5 | 97 % | −3,23 | −0,05 (850) | −0,71 (649), 83 Läufe ohne Ergebnis | −0,51 |
| 1,0 | 91 % | −2,56 | −0,07 (834) | −0,11 (633) | −0,51 |
| 2,0 | 79 % | −2,06 | −0,13 (816) | −0,13 (602) | −0,52 |
| 4,0 | 62 % | −1,73 | −0,19 (800) | −0,20 (568) | −0,51 |
| 8,0 | 43 % | −1,48 | −0,26 (790) | −0,26 (540) | −0,50 |
| 16,0 | 27 % | −1,31 | −0,33 (784) | −0,33 (522) | −0,51 |

**Landmarke wählen, Wahrheit β = −0,5, Wartezeit 4 Tage** (200 Läufe; „Streuung“ = Standardabweichung von β̂ über die Läufe):

| Landmarke τ (Tage) | bleiben | davon behandelt | β̂ (Mittel ± Standardfehler) | Streuung zwischen den Läufen | Überdeckung |
|---|---|---|---|---|---|
| 0,5 | 990 | 116 | −0,176 ± 0,006 | 0,09 | 4 % |
| 1,0 | 942 | 210 | −0,187 ± 0,005 | 0,07 | 1 % |
| 1,5 | 873 | 281 | −0,192 ± 0,005 | 0,07 | 0 % |
| 2,0 | 801 | 329 | −0,190 ± 0,005 | 0,07 | 0 % |
| 2,5 | 734 | 359 | −0,193 ± 0,005 | 0,07 | 0 % |
| 3,0 | 672 | 376 | −0,197 ± 0,005 | 0,08 | 3 % |
| 3,5 | 616 | 382 | −0,200 ± 0,006 | 0,08 | 6 % |
| 4,0 | 567 | 383 | −0,201 ± 0,007 | 0,10 | 10 % |
| 4,5 | 523 | 377 | −0,201 ± 0,008 | 0,11 | 14 % |
| 5,0 | 484 | 369 | −0,195 ± 0,008 | 0,12 | 22 % |
| 5,5 | 450 | 358 | −0,192 ± 0,009 | 0,13 | 25 % |
| 6,0 | 419 | 346 | −0,194 ± 0,011 | 0,15 | 36 % |
| 6,5 | 392 | 334 | −0,194 ± 0,011 | 0,15 | 42 % |
| 7,0 | 367 | 321 | −0,186 ± 0,012 | 0,17 | 50 % |
| 7,5 | 345 | 308 | −0,172 ± 0,013 | 0,18 | 50 % |
| 8,0 | 324 | 294 | −0,166 ± 0,015 | 0,21 | 56 % |

**Stichprobengröße, Wahrheit β = −0,5, Wartezeit 4 Tage** (200 Läufe je Zeile):

| Container n | naiv β̂ | Landmarke Tag 2: β̂ (bleiben) | zeitabhängig β̂ (Überdeckung) |
|---|---|---|---|
| 200 | −1,729 ± 0,014 | −0,200 ± 0,011 (160) | −0,508 ± 0,013 (94 %) |
| 500 | −1,715 ± 0,009 | −0,192 ± 0,007 (401) | −0,504 ± 0,008 (96 %) |
| 1000 | −1,720 ± 0,006 | −0,190 ± 0,005 (801) | −0,507 ± 0,005 (96 %) |
| 2000 | −1,713 ± 0,004 | −0,190 ± 0,003 (1601) | −0,493 ± 0,004 (96 %) |

**Ohne unsterbliche Zeit.** Bei einer Vorab-Zuteilung (die Hälfte der Container wird zufällig am Tag 0 behandelt, unabhängig von der Lebenszeit, keine Wirkung, n = 1000, 200 Läufe) findet die naive Auswertung −0,001 ± 0,004: Wo der Status
am Beginn der Nachbeobachtung zufällig und unabhängig von der Lebenszeit feststeht, entsteht keine Scheinwirkung (gemessen für diese Zuteilung).

## Ehrliche Grenzen

- **Ein Mechanismus.** Der Behandlungsbeginn ist unabhängig von der Lebenszeit ohne Behandlung und vom Zustand des Containers (Exponential-Wartezeit). Wird nach dem Zustand behandelt, kommt ein Störfaktor hinzu; das ist nicht gemessen, auch nicht, ob eine der drei Auswertungen ihn beseitigt.
- **Konstante Wirkung, eine Behandlung, keine Zensierung, keine Anfälligkeit.** Andere Formen der Wirkung oder der Grundrate sind nicht gemessen.
- **Die Landmarke schätzt eine andere Größe als β.** Sie vergleicht „bis τ behandelt“ mit „bis τ nicht behandelt“ unter den Überlebenden bis τ. Ein Abstand zu β ist deshalb kein Fehler der Rechnung allein; die Überdeckung gegen β ist nur ein Maß
  für den Abstand, kein Test der Landmarke gegen ihre eigene Zielgröße.
- **Zwei Landmarken in den Messungen, eine frei in der App.** Wie man τ wählt, zeigt die Demo nicht; die Reihe zeigt nur, dass τ in dieser Welt die Größe des Verlusts kaum, die Streuung und die Zahl der Container aber stark verändert.
- **Nur eine Wartezeit-Verteilung (Exponential)** und eine Grundrate (Lognormal, Median 4 Tage, σ = 0,9); die Reihe über die Wartezeit mittelt nur 100 Läufe je Wartezeit.
- **Cox-Anpassung ist eigener Code** (Breslow); geprüft gegen unabhängige Rechenwege (Schleifen-Cox mit `scipy.optimize.brentq`, die kopierte `cox_fit`), nicht gegen ein Fremdpaket.
- **Die Deutungen** („Verdünnung“, „bei kurzer Wartezeit sind die Unbehandelten die frühen Ausfälle“) sind eigene Lesarten; gemessen ist die Verdünnung (Befund 4), die zweite Lesart nicht.
- **Nur am Rechner angesehen**, nicht auf Mobilgeräten.

## Annahmen, Grenzen und bekannte Probleme (wie in der App)

**Annahmen dieser Demo**
- **Zensierung unabhängig vom Ereignis:** In dieser Welt gibt es keine Zensierung, jeder Container wird bis zu seinem Ausfall beobachtet. Die Annahme wird hier nicht gebraucht und nicht geprüft; mit Zensierung ist nichts gemessen.
- **Die Behandlung beginnt unabhängig von der Lebenszeit ohne Behandlung und vom Zustand des Containers** (Beginn nach einer Exponential-Wartezeit, ohne Blick auf den Container; bei β ≠ 0 hängt die beobachtete Lebenszeit über die Wirkung vom Beginn ab). Wird nach dem Zustand behandelt, kommt ein Störfaktor hinzu; das ist hier nicht gemessen.
- **Die Wirkung ist konstant:** Ab dem Beginn gilt die Rate mal e^β, vorher die Grundrate. Die Grundrate ist lognormal (Median 4 Tage, σ = 0,9). Andere Formen der Wirkung oder der Grundrate sind nicht gemessen.
- **Die Container sind unabhängig voneinander;** es gibt keine unbeobachtete Anfälligkeit und keine weiteren Merkmale.

**Grenzen**
- Eine Behandlung je Container; einmal begonnen, bleibt sie bis zum Ausfall. Abbruch, Wiederholung oder mehrere Behandlungen kommen nicht vor.
- Gemessen sind zwei Landmarken (Tag 2 und Tag 4) in den Messungen der README und eine frei wählbare in der App; wie man eine Landmarke wählt, zeigt die Demo nicht.
- Die Zielstudien-Emulation (Klonen, Zensieren, Gewichten) wird nicht gerechnet; sie ist nur als Suchtreffer genannt.
- Die Cox-Anpassung (Breslow) ist eigener Code, gegen unabhängige Rechenwege geprüft, aber nicht gegen ein Fremdpaket (lifelines, statsmodels).

**Bekannte Probleme und Unsicherheiten**
- Die Zahlen sind Eigenmessungen mit festen Zufalls-Seeds an einer ausgedachten Welt; sie sind keine Literaturwerte. Die Größe des Verlusts der Landmarke (−0,19 statt −0,5) ist Eigenmessung; die Literatur stützt nur die Richtung.
- Die Reihe in Experiment 2 mittelt je Wartezeit über 100 statt 200 Läufe, der Standardfehler des Mittels ist dort entsprechend größer. Läufe ohne Ergebnis (perfekte Trennung bei sehr kurzer Wartezeit) gehen nicht in das Mittel ein; die Tabelle nennt ihre Zahl.
- Nur am Rechner angesehen, nicht auf Mobilgeräten.

## Quellen und Lesestufen

- J. Wang, P. N. Peduzzi, M. Wininger und S. Ma, *Statistical Methods for Accommodating Immortal Time: A Selective Review and Comparison*, arXiv:2202.02369: **im Volltext gelesen**. Landmarke bedingt auf den Status bei τ und verliert Zielpopulation und Präzision;
  die Wahl ist schwierig und subjektiv; ohne die Annahmen einer gültigen Karenzzeit sei sie eher beschreibend als kausal (sinngemäß); zeitabhängige Modellierung gilt als Goldstandard, wenn sie sauber ausgeführt wird. In ihrer Simulation (n = 5 000, wahres log-Hazard-Verhältnis +0,5,
  Landmarken 3, 6, 9 Monate) haben alle Verfahren mit neu gesetztem Nullpunkt negativen Bias, die Landmarke am wenigsten (etwa −0,05 bis −0,17), die Überdeckung der Intervalle liegt bei der Landmarke bei 57 bis 93 % (Spanne der Quelle, hier nicht nach Szenario aufgeschlüsselt).
  **Das stützt die Richtung der Eigenmessung (die Landmarke mindert die Verzerrung, unterschätzt aber den Betrag der echten Wirkung), nicht die Größe**; −0,19 statt −0,50 ist Eigenmessung in einer anderen Welt und kein Literaturwert.
- M. A. Hernán, J. A. C. Sterne, J. P. T. Higgins, I. Shrier und S. Hernández-Díaz, *A structural description of biases that generate immortal time*, Epidemiology (online 4. 11. 2024, DOI 10.1097/EDE.0000000000001808): **nur der Abstract gelesen**.
  Immortal Time entsteht demnach durch Auswahl nach Kriterien, die erst nach der Zuteilung gelten, oder durch Fehlklassifikation; Zielstudien-Emulation verhindert das. Aussagen zur Landmarke im Volltext konnten nicht gelesen werden.
- Zielstudien-Emulation (Klonen–Zensieren–Gewichten): ein weiterer, vollständigerer Ansatz laut Suchtreffern (nicht gelesen): **nur Suchtreffer**.
- **Zum Namen** (eigene Lesart): „Immortal Time“ beschreibt die Auswertung, nicht die Container: Sie sind in dieser Zeit sterblich, es zählt nur so, als wären sie es nicht. Dass jemand den Namen „irreführend“ nennt, steht im gelesenen Abstract nicht
  (im Volltext nicht geprüft); der Mechanismus ist real und hier gemessen.

## Tests

Die Suite hat **129 Tests** (`pytest --collect-only -q tests`); Modell-, Orakel- und Eigenschaftstests laufen in wenigen Sekunden; den größeren Teil der Laufzeit bestimmen der AppTest und die Neuberechnung der Messzellen in `tests/test_claims.py`.

| Datei | Prüft |
|---|---|
| `tests/test_core.py` | kopierte Kernfunktionen (Cox mit Eintritt, Trennung und Singularität als Status, Eingabeprüfung, SplitMix64) gegen die Handbeispiele der Linie |
| `tests/test_scenario.py` | Umkehrfunktion der Normalverteilung, Grundrate (Handwerte, Rundlauf), Zeitumrechnung von Hand (H0(T) = ln 2 + 1/2), ohne Wirkung keine Abhängigkeit von S, gleiche Zufalls-Seeds, gepaarte Welten, Reglerecken |
| `tests/test_model.py` | Handbeispiel in Brüchen (6,0 / 4,5, 7,0 / 7,0, Ränder von τ, zehn Zeilen der aufgeteilten Person-Zeit), Zeilen der drei Auswertungen, schnelle Cox-Anpassung gegen die kopierte `cox_fit` (60 Zufallsdatensätze mit Gleichständen und Eintritt), Status statt stiller Zahlen, Zusammenfassungen |
| `tests/test_oracle_immortal_time.py` | **Orakel:** die drei Auswertungen gegen eine Schleifen-Cox aus den Rohdaten (Status je Container direkt aus T und S, Nullstelle mit `scipy.optimize.brentq`; 2 mal 80 Datensätze, ohne und mit Gleichständen); Erzeuger gegen scipy (Umkehrfunktion, Grundrate, Rückweg der Zeitumrechnung ergibt Exp(1) per Kolmogorov-Smirnov, ohne Wirkung lognormal und unabhängig von S); Simulation mit bekannter Wahrheit als Band (naiv rund −1,28, Landmarke und zeitabhängig bei 0 ohne Wirkung; mit Wirkung zeitabhängig bei der Wahrheit, Landmarke zu schwach), Vorab-Zuteilung ohne Scheinwirkung, zensierte Landmarke findet die Wirkung |
| `tests/test_eigenschaften_immortal_time.py` | Zeitskala und monotone Zeittransformation ändern β̂ nicht, Umsortieren, vertauschte Status kehren das Vorzeichen um, Landmarke bei τ ≈ 0 mit Beginn 0 gleich naiv, ohne Wartezeit keine Scheinwirkung (aber nicht bei kurzer Wartezeit), mehr Container weniger Streuung, stärkere Wirkung kleineres β̂, gleicher Zufalls-Seed |
| `tests/test_edge_cases.py` | n = 1 und 2, Behandlung nach dem Ausfall, alle oder niemand behandelt, τ jenseits aller Zeiten und genau auf einer Zeit, Gleichstände (Breslow), Reglerecken der Welt, sehr kurze Wartezeit meldet Trennung |
| `tests/test_claims.py` | jede Zahl dieser README: Messzellen neu gerechnet gegen die eingefrorene Datei, Tabellen und Prosa-Zahlen gegen die README, Zählungen und Spannen der Aussagen, Annahmen-Text in App und README |
| `tests/test_verdicts.py` | jedes Verdikt der App an handgesetzten Werten (Schwellen, jede Zahl im Text) |
| `tests/test_presets.py`, `test_layout.py`, `test_visualization.py` | Presets und Reglergrenzen; Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Presets, jeder Regler hebt sie auf; Diagramme (Achsen gesperrt, Wahrheit eingezeichnet) |
| `tests/test_app.py`, `test_footer.py`, `test_runtime_imports.py` | AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, Kombination aller Ränder, Experimentregler an den Rändern, Voreinstellung und Preset mit Wirkung), Footer, Laufzeit-Importe gegen `requirements.txt` |

**Zahl → Test.** Zahlen aus den Messungen und aus den Presets: eingefroren in der JSON-Datei und in `tests/test_claims.py` mit dem Code der Demo neu gerechnet (Band 1e-6, Zählungen exakt; die Cox-Anpassung ist iterativ, ein Linux-Lauf wich in der neunten Stelle ab); die Tabellen der README stammen Wort für Wort aus der Datei.
Die Brüche des Handbeispiels: exakt in `tests/test_model.py`. Das „Verhältnis der Mittel“ (38 % und 40 %) und die Spannen der Reihen folgen aus denselben Zahlen (`tests/test_claims.py`). Die Zahlen der Vormessung stehen nur als Vergleich im Abschnitt „Befunde und Korrekturen gegenüber der ersten Messung“.
Nicht im Test: die Deutung „bei kurzer Wartezeit sind die Unbehandelten die frühen Ausfälle“ (eigene Lesart) und die Lesestufen der Quellen.


## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-App: vier Experimente, Annahmen, Formeln |
| `imt_core.py` | Kopie des gemeinsamen Kerns `uz_core.py` der Überlebenszeit-Linie (Stand 2026-11-10): SplitMix64, `check_data`, `cox_fit` mit Eintritt; wird nicht verändert |
| `imt_scenario.py` | Welt und Erzeuger, Umkehrfunktion der Normalverteilung, kumulierte Grundrate |
| `imt_model.py` | die drei Auswertungen als Zeilen, schnelle Cox-Anpassung für ein 0/1-Merkmal, Handbeispiel in Brüchen, Läufe und Zusammenfassungen |
| `imt_evaluation.py` | Formatierung und Verdikte (reine Funktionen) |
| `imt_visualization.py` | Plotly-Abbildungen |
| `imt_constants.py`, `imt_presets.py`, `imt_layout.py` | Regler-Grenzen, Annahmen-Text, Schnellstart, fester Kopf |
| `tools/measure_immortal_time.py` | Messskript; `tools/measure_immortal_time_result.json` die eingefrorenen Messungen |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Zielstudien-Emulation** (Klonen, Zensieren, Gewichten) und Verfahren mit Karenzzeit.
- **Behandlung nach dem Zustand des Containers** (Störfaktor) und alle weiteren Mechanismen des Behandlungsbeginns.
- **Zensierung** und unbeobachtete Anfälligkeit.
- **Mehrere oder abgebrochene Behandlungen**, zeitlich veränderliche Wirkung.
- **Automatische Wahl der Landmarke** und Fremdpaket-Vergleich der Cox-Anpassung.
- **PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Überlebenszeit: zensierte Zeiten lesen](https://sebastianhanisch.net/konzepte-ueberlebenszeit.html).
