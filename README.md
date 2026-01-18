# ACES
ACES - 


PROSJEKTSTYRING: ACES (Adaptive Catch Estimation System)

Tenker dette er nok til kravspesifikasjon.
La og til de mest sentrale delene i backlog, men den må kansje utvides og kombineres med backlogg laget fra før.

Dato: 18.01.2026
Versjon: 1.0 (Master Plan)
Author: Sebastian Tøkje.

DEL 1: KRAVSPESIFIKASJON (Til Rapporten)
Vi benytter MoSCoW-metoden for å prioritere funksjonalitet. Dette sikrer at vi leverer en fungerende prototype (MVP) innen fristen, samtidig som vi har klare mål for høyere karakteroppnåelse og videre kommersiell utvikling.
3. Kravspesifikasjon (MoSCoW)
Vi har prioritert funksjonaliteten for å sikre en leverbar MVP (Minimum Viable Product).
🔴 P0 - MUST HAVE (Kritisk for projektet)
Robust Bildeakvisisjon: Global Shutter-kamera i vanntett hus (IP67) som leverer skarpe bilder av transportbåndet.
Deteksjon: AI-modell (YOLOv11/26-Seg) som skiller Torsk og Sei med helst en >90% nøyaktighet.
Telle-logikk: Implementering av Line Crossing med hysterese (to linjer) for å unngå dobbelttelling når båndet stopper eller fisken sklir tilbake.
Støyfiltrering: Systemet må trenes på "Negative Samples" (tomt bånd med blod/vann) for å unngå falske positiver.
Lokal Logging: Data (Tid, Art, Antall) lagres kontinuerlig til CSV.
🟡 P1 - SHOULD HAVE 
Instans-segmentering: Bruk av masker (polygoner) i stedet for bokser for å håndtere fisk som ligger oppå hverandre (okklusjon).
Kapteinens Dashboard: Grafisk grensesnitt (GUI) som viser live video og tellere.
Day-to-Night Loop: Automatisert lagring av usikre bilder og script (night_train.py) for lokal re-trening.
Black Box Opptak: Video lagres som .mkv (ikke mp4) for å tåle strømbrudd uten filkorrupsjon.
Kill Zone (ROI): Ignorering av deteksjoner i bildekanten for å øke presisjon.
Gjennkjenne andre arter: Modellen burde kunne gjennkjenne andre arter som, lyr, hyse. Og evt andre arter om datagrunnlaget gir modellen mulighet til å lære det.
🟢 P2 - COULD HAVE (Ved god tid)
Vekt-estimering: Algoritme som omgjør maske-areal til vekt basert på en justerbar faktor.
ArUco Kalibrering: Automatisk skalering av piksler basert på en referanse-markør.
⚪ P3 - WON'T HAVE (Fremtidige Krav)
Disse kravene ligger utenfor Bachelor-oppgavens omfang.
Edge-Hardware: Porting av software fra Laptop til NVIDIA Jetson Orin Nano.
Cloud Connectivity: Automatisk opplasting/nedlasting av data, bilder modeller til en skybasert flåte-database.



DEL 2: MASTER BACKLOG (Til GitHub)
Opprett disse som Issues i GitHub-repoet ditt.
🛠️ Hardware (Må være ferdig i Fase 1)
[HW] Innkjøp (P0): Bestill Arducam AR0234, IP67 boks, M25 nippel, Silica Gel, Aktiv USB-kabel.
[HW] Bygg Kameraboks (P0): Bor hull, lim vindu (Tec7), monter nippel med vulkteip.
[HW] Fokus-låsing (P0): Koble til PC, still inn fokus på 1 meter, og lim/teip fast fokusringen.
[HW] Stress-test (P0): "Dusj-testen" (10 minutter) med papir inni for å sjekke lekkasjer.
🐟 Data Factory (Fundamentet)
[DATA] Opptak Tur 1 (P0): Sikre 8 timer råvideo fra båten/haling.
[DATA] Cherry Picking (P0): Trekk ut 500??? gode bilder av fisk (varier bilde vinkle for en mer robust mode  ”vri bilde 90 grader osv..”) med script?
[DATA] Negative Samples (P0): Trekk ut 50 bilder av tomt bånd (med blod/rot) og lagre som tomme labels.
[DATA] Labeling V1 (P0): Tegn polygoner på alle 550 bildene i Roboflow/LabelMe. ?? Usiker på hva verktøy som er best. Må ha riktig format til YOLO modell, men tror det er standart format?
💻 Software (Kjernen)
[SW] Setup (P0): Installer Python, PyTorch og Ultralytics (YOLOv11/v5/v26?) på Laptop.
[SW] Trening V1 (P0): Tren første modell. Verifiser at den finner fisk på en test-video.
[SW] Black Box Recorder (P1): Skriv koden som lagrer video kontinuerlig til .mkv (ikke mp4).
[SW] Telle-logikk (P0): Implementer ByteTrack + Hysterese (Linje A/B) i Python.
[SW] ROI / Kill Zone (P1): Legg inn if y < 100: continue for å ignorere kantene.
[SW] Active Learning Filter (P1): Implementer if 0.3 < conf < 0.7: save_image(). og ikkje tell?
[SW] Natt-script (P1): Lag night_train.py for å automatisere re-trening om natten.
[SW] Kapteinens Dashboard (P1): Vis store tall på skjermen med cv2.putText.
[SW] Vekt-algoritme (P2): Implementer Vekt = Maske_Areal * Faktor. usikker?
🧪 Testing & Bevis (Vitenskapen)
[TEST] Felt-test Alpha (P0): Første tur på båten. Verifiser at hardware overlever.
[TEST] Golden Hour (P1): Manuelt tell antall fisk i 1 time video ("Fasiten").
[TEST] Sammenligning (P1): Kjør AI på samme time. Regn ut nøyaktighet i %.
[TEST] Nøyaktighets-graf (P1): Bevis at modellen ble bedre fra Dag 1 til Dag 3 (Active Learning).
📄 Dokumentasjon (Løpende arbeid)
[DOCS] Daglig Logg (P0): Skriv i LAB_DAGBOK hver arbeidsdag.
[DOCS] Beslutningslogg (P0): Dokumenter valg (Global Shutter, YOLO valg, etc).
[DOCS] Skriv Innledning (P0): ().
[DOCS] Skriv Metode (P0): ().
[DOCS] Skriv Kommersiell Plan (P3): Beskriv fremtidsvisjonen i "Videre Arbeid"-kapittelet. ?? usikker..



