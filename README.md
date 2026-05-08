ACES Readme 

Filstruktur, input/output og oppstart

ACES startes fra prosjektets rotmappe (`ACES/`) med følgende kommando:

```bash
streamlit run apps/dashboard/app.py
Hovedfilen for applikasjonen ligger i: ACES/apps/dashboard/app.py
Prosjektets Python-avhengigheter er definert i: ACES/requirements.txt

--------------------------------------------------
Hovedstruktur i prosjektet:

•	ACES/apps/dashboard/
Inneholder Streamlit UI, sider, services, managers og integrasjonslogikk mellom frontend og backend.
•	ACES/data/
Inneholder inputdata, historikk, review-data og datasett brukt av systemet.
•	ACES/data/input/
Mappe for videofiler og inputdata brukt under kjøring. Nye testvideoer plasseres her. Nåværende eksempelvideo ligger også i denne mappen.
•	ACES/data/sample/
Inneholder et lite eksempel-datasett brukt for testing av pipeline med bildedata.
•	ACES/outputs/
Inneholder output fra kjøringer, logger, track-resultater og trenede modellfiler.
•	ACES/outputs/weights/best.pt
Nåværende trenede modell som brukes av systemet.
•	ACES/scripts/
Inneholder scripts for full trening, night training og hjelpeverktøy relatert til trening.
•	ACES/src/
Inneholder hovedlogikken for deteksjon, tracking, telling, active learning og retrening.

---------------------------------------------
1.Runtime-pipelinen

Runtime pipeline i ACES prosesserer input sekvensielt gjennom deteksjon, sporing, telling og håndtering av sesjonsdata.
Input gis som videostrøm eller bildemappe. Frames leses ett og ett og sendes videre til tracking-steget.
Deteksjon og sporing håndteres samlet i src/vision/track/tracker.py ved bruk av Ultralytics model.track() med en YOLOv11 segmenteringsmodell og ByteTrack. For hvert frame returneres en liste med objekter som inneholder bounding box, class_id, confidence, senterpunkt og en persistent track_id. Track_id gjør det mulig å følge samme fisk over flere frames.
En separat deteksjonsmodul finnes i src/vision/detect/ (FishDetector). Denne brukes til testing og evaluering av modellen, og reflekterer at deteksjon er en egen logisk komponent. I runtime-pipelinen utføres deteksjon internt i tracking-steget for en enklere og mer stabil implementasjon.
Tracked objects sendes videre til tellemodulen i src/vision/count/counter.py. Telling er basert på en linjekryssingsmetode. For hvert objekt lagres posisjonshistorikk, og posisjonen klassifiseres som “before”, “middle” eller “after” relativt til en virtuell linje. En fisk telles én gang når den følger et gyldig kryssingsmønster (before → middle → after). Hvert track_id kan kun telles én gang, og gamle spor fjernes automatisk.
Pipelinen orkestreres i apps/dashboard/services/home_manager.py. Denne komponenten håndterer input, initialiserer tracker og counter, og kjører prosesseringen frame for frame. For hvert steg kalles tracker, resultatene sendes til counter, sesjonsdata oppdateres, og Active Learning trigges ved behov. Resultatene gjøres deretter tilgjengelige for UI-laget.
Prosesseringen skjer sekvensielt per frame for å bevare tidsrekkefølgen som er nødvendig for tracking og telling. Pipelinen kjøres normalt i en bakgrunnstråd for å unngå blokkering av brukergrensesnittet.

-----------------------------------------------
2. Active learning pipeline

Active Learning & Continuous Retraining For å sikre at systemet kontinuerlig kan tilpasse seg nye fartøy og miljøer (Domain Adaptation), har vi implementert en Active Learning-loop i inferens-koden. Mens modellen (V1.1) analyserer videostrømmen fra transportbåndet, evaluerer den kontinuerlig sin egen usikkerhet. Dersom modellen detekterer en fisk, men konfidens-scoren faller innenfor en definert "tvil-sone" (f.eks. mellom 30 % og 70 %), trigger systemet en lagringsfunksjon. Den spesifikke framen lagres automatisk i en egen dedikert mappe (/active_learning_samples). Fordelen med denne tilnærmingen: Fremfor å manuelt lete gjennom timesvis med video etter feil, fungerer systemet som en intelligent "Hard Example Miner". Den plukker kun ut de bildene der den sliter med overlappende fisk, vanskelige skygger eller fremmedlegemer. Disse filene kan deretter enkelt pre-labeles ved hjelp av eksisterende modell, justeres manuelt i X-AnyLabeling, og flettes inn i datasettet for neste treningsløp (night_train.py). Dette gjør systemet selvlærende og eksponentielt mer robust over tid, med minimalt manuelt arbeid.

-----------------------------------------------
3. Data structure

ACES lagrer fangstdata som strukturerte JSON-filer organisert i tur og økt. En tur representerer en rapporteringsperiode, typisk tiden båten er til havs, mens en økt er en aktiv telleperiode der systemet prosesserer videostrøm og registrerer fangst. En tur kan inneholde flere økter.
Dataflyten går fra runtime-pipelinen til SessionService, videre til lagring som JSON-filer, og deretter til HistoryService som leser og aggregerer data for visning i brukergrensesnittet. Runtime-pipelinen genererer tellinger per frame, SessionService oppdaterer aktiv økt fortløpende, og data lagres kontinuerlig til disk gjennom autosave. HistoryService leser lagrede økter og grupperer disse per tur.

------------------------------------------------
##Data lagres i følgende struktur: 
●	data/history/sessions/ inneholder individuelle økter som JSON-filer
●	data/history/trips/ inneholder metadata for turer
●	data/history/active_trip.json representerer aktiv tur.
●	SessionService håndterer opprettelse, oppdatering, autosave og avslutning av økte
●	SessionManager skriver sesjonsdata til disk
●	TripService håndterer opprettelse og oppdatering av turer, samt koblingen mellom tur og økter.
●	HistoryService aggregerer økter til tur-nivå statistikk
●	HistoryManager leser lagrede JSON-filer fra disk.
Per økt lagres:
●	 start- og sluttid, 
●	varighet, 
●	artsfordeling (species_counts), 
●	totalt antall fisk, 
●	antall usikre deteksjoner, 
●	antall objekter sendt til gjennomgang, 
●	antall korreksjoner
●	status
Økter autosaves kontinuerlig under kjøring for å redusere risiko for datatap. Hver økt lagres som en egen JSON-fil, og kobles til riktig tur via trip_id. Historikk rekonstrueres ved å lese og aggregere lagrede økter.

##Art og vikt 

ACES benytter et felles system for håndtering av arter og vektberegning som brukes på tvers av hele applikasjonen. Systemet sikrer at alle deler av løsningen benytter samme artsregister og vekter, og fungerer som en felles kilde til sannhet.
Artsinformasjon lagres i src/common/species.py, som definerer mapping mellom class_id og artsnavn (CLASS_NAMES) samt omvendt mapping (NAME_TO_CLASS_ID). Denne filen brukes av tracking, review, trening, historikk og UI, og sikrer konsistent håndtering av arter gjennom hele systemet.
Nye arter kan legges til dynamisk via SettingsService (apps/dashboard/services/settings_service.py). Når en ny art legges til, oppdateres species.py, og runtime_config.json oppdateres samtidig med en standard snittvekt. Artsregisteret lastes deretter på nytt ved hjelp av importlib.reload(), slik at endringen blir tilgjengelig i systemet uten restart.
I review-flyten (apps/dashboard/services/review_service.py) brukes det samme artsregisteret for å vise og endre art på usikre deteksjoner. Oppdatert artsliste hentes dynamisk, slik at nye arter umiddelbart blir tilgjengelige for brukeren.
Vektberegning håndteres av WeightManager (apps/dashboard/services/weight_manager.py). Systemet benytter en konfigurerbar snittvekt per art, lagret i configs/runtime_config.json. Basert på antall fisk per art beregnes estimert vekt som antall multiplisert med snittvekt. Resultatet brukes i både runtime-visning og historikk.
Art- og vektsystemet er dermed tett integrert med flere deler av applikasjonen, men samlet gjennom en delt datastruktur og felles konfigurasjon.

----------------------------------------------
4. UI-lag

## UI, services og managers

Dashboardet er strukturert med en separasjon mellom UI-sider, service-klasser og manager-klasser.

- **UI-sider** inneholder Streamlit-layout og brukerinteraksjon.
- **Services** fungerer som et bindeledd mellom UI og backend-logikk.
- **Managers** inneholder orkestrering, domenelogikk og filhåndtering.

Denne strukturen holder UI-laget enkelt og gjør systemet lettere å utvide, teste og vedlikeholde.

### Viktige UI-filer

- `apps/dashboard/app.py`  
  Hovedinngangspunkt for Streamlit-applikasjonen. Setter sidekonfigurasjon, laster navigasjon og håndterer routing mellom dashboard-sidene.

- `apps/dashboard/components/top_nav.py`  
  Renderer toppnavigasjon og statusindikatorer brukt på tvers av dashboardet.

- `apps/dashboard/pages/home_page.py`  
  Hoved-/runtime-side. Viser aktiv tur og økt, live telledata, estimert vekt og start/stopp-kontroller for telling.

- `apps/dashboard/pages/review_page.py`  
  Gjennomgangsside for usikre deteksjoner. Viser bilder i review-køen og lar brukeren godkjenne, avvise, endre art eller sende objekter til land/admin-gjennomgang.

- `apps/dashboard/pages/history_page.py`  
  Historikkside. Viser lagrede turer og økter, inkludert artsfordeling og estimert vekt. Støtter også lokal eksport av historiske data.

- `apps/dashboard/pages/settings_page.py`  
  Innstillingsside. Håndterer systemkonfigurasjon som inputkilde, valgt modell, Active Learning-thresholds, arter og snittvekter.

### Services

- `apps/dashboard/services/home_service.py`  
  Kobler Home-siden til runtime-systemet. Starter og stopper prosessering og kjører HomeManager-loopen, vanligvis i en bakgrunnstråd slik at UI forblir responsivt.

- `apps/dashboard/services/session_service.py`  
  Håndterer livssyklusen til aktive økter. Oppretter nye økter, oppdaterer artsfordeling og usikre deteksjoner, utfører autosave og avslutter økter ved stopp.

- `apps/dashboard/services/review_service.py`  
  Adapter mellom Review-siden og ReviewManager. Forbereder review-data for UI, laster artsdefinisjoner på nytt ved behov og videresender brukerhandlinger som godkjenning, avvisning og endring av art.

- `apps/dashboard/services/history_service.py`  
  Leverer historikkdata til UI ved å lese lagrede økter og turer gjennom historikklaget.

- `apps/dashboard/services/settings_service.py`  
  Leser og skriver systeminnstillinger. Brukes av dashboard og runtime-system for tilgang til konfigurasjon som valgt modell, input-paths, thresholds og artsvekter.

- `apps/dashboard/services/trip_service.py`  
  Håndterer opprettelse av turer og aktiv tur-status. Sikrer at økter kobles til riktig aktiv tur.

### Managers

- `apps/dashboard/services/home_manager.py`  
  Hovedorkestrator for runtime i dashboardet. Kobler sammen inputhåndtering, tracking, telling, sesjonsoppdateringer, Active Learning og vektoppsummering. Støtter både video-input og bildemapper, henter valgt modell og input-paths fra settings, håndterer start/stopp-status og prosesserer frames gjennom `step()`.

- `apps/dashboard/services/session_manager.py`  
  Ansvarlig for lagring av sesjonsdata som JSON-filer på disk. Brukes av SessionService til autosave og permanent lagring av økter.

- `apps/dashboard/services/review_manager.py`  
  Håndterer filbaserte operasjoner i review-køen. Flytter eller oppdaterer review-objekter når brukeren godkjenner, avviser, endrer art eller sender objekter til land/admin-gjennomgang.

- `apps/dashboard/services/history_manager.py`  
  Leser lagrede økt- og turfiler og aggregerer disse til datastrukturer brukt av historikksiden.

- `apps/dashboard/services/weight_manager.py`  
  Beregner estimert fangstvekt basert på artsfordeling og konfigurerte snittvekter per art. Brukes både i live-visning og historiske oppsummeringer.

### Shared state og sesjonshåndtering

- `apps/dashboard/state.py`  
  Lagrer delt runtime-state for Streamlit-applikasjonen, inkludert aktiv økt og delte service-/manager-instanser. Dette er nødvendig fordi Streamlit kjører side-scriptet på nytt under interaksjon.

En sesjon, eller **økt**, representerer én aktiv telleperiode. Når brukeren starter telling oppretter `SessionService` en ny økt og kobler den til aktiv tur. Under runtime prosesserer `HomeManager` frames og oppdaterer økten gjennom `SessionService`. Økten lagrer blant annet artsfordeling, totalt antall fisk, usikre deteksjoner, review-objekter og tidsstempler.

Sesjonsdata autosaves kontinuerlig mens systemet kjører og lagres permanent når brukeren stopper økten. Dette gir dashboardet live-state under drift samtidig som data bevares for senere historikk og eksport.




