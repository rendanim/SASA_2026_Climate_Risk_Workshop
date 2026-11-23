<!--
All text shown in the app lives here, so wording can change without touching code.

Format: each block starts with a line "## <key>" and runs until the next "## ".
Words in {curly_braces} are replaced with numbers the app computes live; an unknown
{name} is shown as-is. LaTeX goes between $ signs.
-->

## home.intro
Africa has some of the sparsest rain-gauge networks in the world, and the newest weather models have only a few months of forecasts to learn from. This app works through one real answer to that problem: take **AIFS-Single**, ECMWF's machine-learning weather model, and turn its single rainfall number into **calibrated probabilities** using **isotonic distributional regression (IDR)**, checked against **GPM IMERG** satellite rainfall.

Every lab follows the same rhythm:

1. **Predict first.** Commit to an answer before touching anything.
2. **Play.** Move a control and watch the plot change.
3. **Explain simply.** A short plain-language explanation, with the maths folded away.
4. **Check yourself.** A discussion question, with an answer built from the numbers on your screen.

## home.pipeline
**The workshop in one line:** AIFS says how much rain to expect in each 0.1° grid box → **IDR**, fitted per climate zone, turns that number into a full probability distribution → we **score** it with the CRPS against IMERG and against climatology → we turn the probabilities into **warnings and risk indices** and ask who benefits.

## home.setup
**The question.** Does AIFS-Single add useful, trustworthy information about tomorrow's rain beyond what climatology already tells us, once its errors are corrected?

**The ingredients.**

- **Forecast:** AIFS-Single 00 UTC runs, 24-hour rainfall, regridded from 0.25° to IMERG's 0.1° grid. Lead day 2 is the rain 24–48 h after the run starts; we use lead days 2–5.
- **Truth:** GPM IMERG daily rainfall, a satellite-based estimate. It is not a rain gauge, and it has its own errors.
- **Reference:** {clim}. Any forecast worth using must beat this.

**The fair test.** Models learn from **{train}** and are judged on **{test}**, days they have never seen.

**Pooling.** Each climate zone gets one model, fitted on pairs from up to {n_zone} grid boxes in the zone, thinned to at most {n_pairs} pairs. The app keeps those exact pairs, plus {n_test} random held-out pairs per zone and lead day, so it can refit and re-score live.

## home.provenance
- Headline scores, maps, zone tables, daily curves and the reliability counts at the notebook's 10 mm/day threshold are the notebook's own numbers, computed on **every** land grid box and test day when the data bundle was built.
- Everything you change with a control (thresholds, subsets, pooling choices, warning rules) is recomputed live on the stored **samples** of training and test pairs. Zone samples are weighted by each zone's number of valid test cells when pooled over Africa.
- The zone models are refitted from exactly the pairs the notebook used, with the same random seed, so they are the notebook's models.

## lab1.intro
Before any statistics, look at what we have to learn from. Two problems define forecasting in data-scarce settings: **short records** (AIFS is new, so we have under a year of forecasts) and **noisy truth** (IMERG is a satellite estimate, and much of Africa has few gauges to correct it).

## lab1.predict.q
AIFS forecasts a completely dry day (0 mm) for a grid box in zone **{zone}**. How often do you think IMERG still records at least 1 mm there?

## lab1.predict.a
In the training data it happened on **{p}** of {n} dry forecasts. A forecast of "0 mm" is not a promise of a dry day. That is the first reason to replace the single number with a probability.

## lab1.budget.caption
A single grid box sees only a handful of heavy-rain days in a year, far too few to estimate a probability of heavy rain. Pooling boxes with a similar climate ("space for time") multiplies the sample, at the price of assuming those boxes behave alike.

## lab1.zones.explain
**Why climate zones?** Rain behaves very differently in a desert, a savannah and a rainforest. Grid boxes in the same Köppen–Geiger zone share a broadly similar rain climate, so we pool their (forecast, observation) pairs and fit **one calibration model per zone**. Every box in the zone then uses that model, including boxes never used in training.

This is the central trade-off for data-scarce forecasting: **pool more** and the model sees more examples, especially of rare heavy rain; **pool less** and the model is more specific to local conditions. Lab 4 measures both sides.

## lab1.zones.maths
For zone $k$ with grid boxes $\mathcal{G}_k$, the training set is $\{(x_{d,g}, y_{d,g}) : g \in S_k, d \in \mathcal{D}_{\text{train}}\}$ with $S_k \subset \mathcal{G}_k$ a random sample of up to 1,000 boxes, thinned at random to at most 50,000 pairs. The model is the same for every $g \in \mathcal{G}_k$. Pairs from the same day are strongly correlated, so the *effective* sample size is much smaller than the number of pairs.

## lab1.scatter.explain
**Left:** each training pair is one grid box on one day. The colour shows how many pairs fall in each cell (log scale), and the dashed line is a perfect forecast. The correlation across all pairs is **r = {r}**. **Right:** how often each source reports rain.

- AIFS reports at least 1 mm on **{f1a}** of days against IMERG's **{f1o}**. Like most models, it tends to spread rain too widely as drizzle.
- At 50 mm or more: AIFS **{f50a}**, IMERG **{f50o}**. Models often smooth out the heaviest falls.

Both are systematic errors, which is good news: systematic errors can be learned and removed.

## lab1.scatter.maths
Axes use $\log_{10}(1 + \text{mm/day})$ so that dry, light and heavy rain are all visible. $r$ is Pearson's correlation over all pairs; it mixes skill at telling wet days from dry days with the differences between places.

## lab1.season.explain
The training set covers about {n_months} months, roughly one cycle of each region's rainy seasons. A model trained on one year has seen only one version of each season: a wet one or a dry one, an El Niño year or not. This is why short records make calibration uncertain, and why Lab 4 asks how skill degrades with less data.

## lab1.check.q
A national meteorological service wants to calibrate AIFS for heavy rain at one station, using only that station's matched record. Using the numbers on this page for zone {zone}, explain why that is hard, and what pooling buys.

## lab1.check.a
One grid box has about **{n_days}** training days, and only about **{heavy_one}** of those had at least 50 mm. No method can estimate a heavy-rain probability from one or two events. Pooling the zone gives **{n_pairs}** pairs including **{heavy_pool}** heavy-rain pairs: enough to see how often AIFS's heavy forecasts verify. The cost is the assumption that the pooled boxes share one forecast-error behaviour. In practice a service could pool by zone, then check skill station by station.

## lab2.intro
AIFS gives **one number** per grid box, such as "12 mm tomorrow". The rain that actually falls could be 0 mm or 40 mm. **Isotonic distributional regression (IDR)** learns, from past pairs, the whole range of outcomes that followed each forecast amount. It imposes just one rule: **a wetter forecast must never make dry outcomes more likely**. No distribution family is assumed, there are no tuning parameters, and the method cannot predict negative rain.

## lab2.predict.q
As the AIFS forecast gets wetter, what should happen to the probability that the observed rain stays **at or below 10 mm**?

## lab2.predict.a
It should go **down**, or at least never go up. That is the *isotonic* (order-keeping) rule, and IDR enforces it. Watch the blue curve below: it can only fall as you move right.

## lab2.threshold.explain
Pick a threshold *t*. The **grey dots** group the training days by their AIFS forecast and show the share of days that stayed at or below *t* (larger dots mean more days). The **blue curve** is IDR's estimate of the same probability. It is the best fit to the data among all curves that never increase.

Change *t* and you get a different curve. **IDR is this, done for every threshold at once.** Stacked together, the curves give a full probability distribution for any forecast.

## lab2.threshold.maths
For each threshold $t$, IDR solves an antitonic least-squares problem:
$$\hat F(t \mid \cdot) = \arg\min_{F \text{ non-increasing in } x} \sum_i \big(F(t \mid x_i) - \mathbf 1\{y_i \le t\}\big)^2 .$$
Solving it at every observed $t$ minimises the mean CRPS over all distributions that respect the order (Henzi, Ziegel & Gneiting, 2021). The app, like the notebook, works on 0.1 mm/day levels and interpolates linearly between training forecast values.

## lab2.pava.explain
How does IDR force the curve downward? With **pool adjacent violators (PAVA)**. Walk from the driest forecast to the wettest. Whenever a wetter forecast shows a *higher* share of days at or below *t* than its neighbour, the rule is broken, so **pool them** and replace both with their weighted average. Repeat until nothing breaks the rule. Step through it on 30 training days.

## lab2.cdf.explain
**Left:** the predicted distribution for six forecast amounts. At any rain amount on the horizontal axis, a curve's height is the probability that the observed rain is at or below that amount. The curve's starting height is the probability of a completely dry day. As the forecast gets wetter, the curves move right, as the isotonic rule requires. Even a forecast of 0 mm usually keeps a real chance of rain.

**Right:** the calibrated median and 10–90% range against the raw forecast. Where the blue band sits below the 1:1 line, AIFS tends to over-forecast; where it sits above, AIFS tends to under-forecast. The band's width is the honest uncertainty that the single number hides.

## lab2.cdf.maths
For a new forecast $x$ between training values $x_{(j)} \le x \le x_{(j+1)}$:
$\hat F(t \mid x) = (1-w)\hat F(t \mid x_{(j)}) + w \hat F(t \mid x_{(j+1)})$, with $w = (x - x_{(j)})/(x_{(j+1)} - x_{(j)})$. Beyond the largest training forecast the distribution stays fixed: **IDR does not extrapolate**, which matters for record-breaking rain.

## lab2.check.q
A district officer reads "AIFS: 0 mm tomorrow" and cancels flood preparations. Using zone {zone}, what would the calibrated forecast have told them instead?

## lab2.check.a
For a 0 mm AIFS forecast, the calibrated model gives a **{p_dry0}** chance of less than 1 mm, so roughly **{p_wet0}** of such days still bring measurable rain. For a 20 mm forecast, the calibrated median is about **{q50_20} mm**. The single number is a reasonable central guess but says nothing about the risk; the distribution says how often to expect a surprise.

## lab3.intro
How do you score a forecast that gives a whole distribution against one observed value? And how do you compare it fairly with a forecast that gives only one number? The **continuous ranked probability score (CRPS)** does both, in the units of the rainfall itself.

## lab3.predict.q
Raw AIFS is a sharp, confident single number. Climatology is a vague but honest spread. Which do you expect to score better on average, and what would change after calibration?

## lab3.predict.a
In the notebook, **raw AIFS scores worse than climatology** at every lead day: a single number that is confidently wrong is penalised heavily. After calibration, AIFS **beats** climatology. The information was in the forecast all along; the raw number just expressed it badly. The headline results below show the size of each effect.

## lab3.crps.explain
The black step is what happened: 0 below the observed amount, 1 above it. The CRPS adds up the squared gap between a forecast's curve and that step, across every rain amount, so it is measured in **mm/day**: lower is better.

- **Raw AIFS** is a single step at its forecast. Its CRPS is exactly the absolute error $|x - y|$.
- **The calibrated forecast** hedges, so it is never badly wrong but rarely perfect.
- **Climatology** ignores the forecast entirely.

Try a good forecast (forecast close to the observation), then a miss (forecast 20, observed 0). Sharpness pays off only when it is right.

## lab3.crps.maths
$$\mathrm{CRPS}(F, y) = \int_{-\infty}^{\infty} \big(F(z) - \mathbf 1\{y \le z\}\big)^2\, dz,$$
the Brier score for "rain ≤ $z$" integrated over every threshold $z$. For a point forecast $F = \mathbf 1\{x \le z\}$ it reduces to $|x - y|$. The skill score is $\mathrm{CRPSS} = 1 - \mathrm{CRPS}/\mathrm{CRPS}_{\text{clim}}$, where 0 means no better than climatology and 1 means perfect.

## lab3.decomp.explain
The notebook writes the raw forecast's error as

$$\mathrm{CRPS}_{raw} = \underbrace{(\mathrm{CRPS}_{raw} - \mathrm{CRPS}_{cal})}_{\text{MCB}} - \underbrace{(\mathrm{CRPS}_{clim} - \mathrm{CRPS}_{cal})}_{\text{DSC}} + \underbrace{\mathrm{CRPS}_{clim}}_{\text{UNC}}$$

For lead day {lead}: **UNC = {unc}** mm/day is the difficulty of the problem, which no forecast controls. **MCB = {mcb}** is the error that calibration removes. **DSC = {dsc}** is what the forecast knows beyond climatology.

**DSC matters most for operational value.** Calibration can always remove MCB, but only a forecast with real information has DSC > 0. A raw forecast can look useless (negative CRPSS) while carrying genuine discrimination.

## lab3.decomp.maths
$\mathrm{CRPSS}_{raw} = (\mathrm{DSC} - \mathrm{MCB}) / \mathrm{UNC}$ and $\mathrm{CRPSS}_{cal} = \mathrm{DSC}/\mathrm{UNC}$. Here "cal" is the IDR forecast fitted on the **training period**, which is the honest choice for a forecast you could actually issue. In the formal decomposition (Arnold et al.; Dimitriadis et al.), the recalibration is fitted on the evaluation data itself, which guarantees MCB ≥ 0 and DSC ≥ 0. The expander below compares the two.

## lab3.insample.explain
Fitting IDR on the {n} test pairs of zone {zone} themselves gives the best calibration any monotone method could have achieved on these days. Its CRPS is lower than the honest out-of-sample forecast's, so its MCB is larger and its DSC larger. The gap between the two rows measures what was lost by learning from a different period. With short records and a changing climate, that gap is part of the uncertainty.

## lab3.daily.explain
The three forecasts day by day, averaged over all of Africa (7-day running mean). Calibration helps most in the rainy seasons, when there is most rain to get wrong. The gap between the climatology line and the calibrated line is the skill that AIFS adds.

## lab3.check.q
Answer the first workshop question for lead day {lead}: what are the three components of the decomposition, and which matters most for operational utility? Use the numbers on this page.

## lab3.check.a
**UNC** (climatology's CRPS) measures how hard the problem is. **MCB = {mcb} mm/day** is error that calibration removes. **DSC = {dsc} mm/day** is information beyond climatology. Raw AIFS has CRPSS **{raw_s}** because its miscalibration outweighs its discrimination; calibrated AIFS reaches **{cal_s}**. DSC is the most important term for operational utility: calibration can always fix MCB, but only a forecast with DSC > 0 is worth issuing.

## lab4.intro
Every number so far came from one training year and one test period. How much should we trust it? This lab measures uncertainty three ways: **how skill changes with less data**, **how the pooling choice matters**, and **how wide the confidence interval really is** when neighbouring grid boxes share the same weather.

## lab4.predict.q
AIFS is new, so we have under a year of forecasts. If we had only **one month** of AIFS runs for zone {zone}, would the calibrated forecast still beat climatology?

## lab4.predict.a
Play A below gives the answer for your zone. Typically skill drops, and varies much more from draw to draw, when training covers only a few weeks: the model has seen only one part of the seasonal cycle and few heavy-rain days. A model trained in the dry season can do worse than climatology in the wet season.

## lab4.lc.time.explain
**Time** is the scarce resource for a brand-new model. Each point trains on a random stretch of consecutive weeks and scores all 2026 test days. With a few weeks, the model sees one season only, so the spread between draws is large: the result depends on *which* weeks you happened to have. That spread is the uncertainty a forecaster faces when a new model arrives.

## lab4.lc.space.explain
**Space** is the scarce resource when you calibrate from a few stations. Each point pools pairs from only *N* grid boxes. With very few boxes the model has seen few heavy-rain days and is noisy; adding boxes helps until extra boxes add little new information, because boxes on the same day share the same weather.

## lab4.lc.pairs.explain
**Pairs** at random is the optimistic case: the same number of pairs, spread over all days and boxes. Compare it with *time* and *space*: for the same number of pairs, random pairs carry more independent information than a short or local record. Correlated data are worth less than their count.

## lab4.lc.maths
For each size, 5 random subsets of the zone's stored training pairs are drawn and IDR is refitted on each. All fits are scored on the same 5,000 held-out 2026 pairs from the zone, as CRPSS against the IMERG climatology. The band is the range over the 5 draws; the dotted line is the notebook's model, fitted on all stored pairs.

## lab4.pool.explain
Three choices of how widely to pool, each scored on the same held-out pairs per zone:

- **Local:** only {n_local} grid boxes, about {n_local_pairs} training pairs, roughly what a handful of stations give in a year.
- **Per climate zone:** the notebook's choice.
- **All of Africa:** one model from {n_africa} pairs, sampled in proportion to each zone's area.

The per-zone model is best or tied-best in **{best_zone}** of {n_zones} zones, the all-Africa model wins in **{best_africa}**, and the local model in **{best_local}**. Pooling too little leaves too few examples; pooling too much mixes rain climates whose forecast errors differ. The best choice depends on how much data you have and how different the regions are.

## lab4.real.explain
**Left: how sure are we?** The estimate rests on {n_days} test days. Rain on neighbouring grid boxes on the same day is strongly linked, so resampling individual pairs (grey) pretends there are thousands of independent cases and gives a falsely narrow interval. Resampling **whole days** (blue) respects that dependence and gives the honest interval.

**Right: is the skill real?** Shuffling the AIFS values before fitting breaks any link between forecast and outcome. The shuffled models score around **{shuffled}**, close to the training-period climatology (**{train_clim}**), while the real model scores **{real}**. Calibration cannot create skill; it can only reveal skill that the forecast already has.

## lab4.real.maths
Day-block bootstrap: draw test days with replacement, keep all of each day's pairs, and recompute $1 - \sum \mathrm{CRPS}_{cal} / \sum \mathrm{CRPS}_{clim}$; 400 replicates. The shuffled control permutes $x$ in the training pairs, refits IDR and re-scores (20 permutations). Its skill is that of a climatology learned from one training year, which is usually worse than a 21-year climatology.

## lab4.check.q
Workshop question 4: AIFS is a new model with only a short record available for recalibration. What can we do to make sure there are enough samples to train on? Use the three experiments on this page for zone {zone}.

## lab4.check.a
- **Pool in space** ("space for time"): borrow pairs from grid boxes with a similar rain climate, such as a Köppen–Geiger zone. Play B shows that this usually beats both very local and continent-wide models.
- **Pool across lead days or seasons** where errors behave alike, and refit as new forecasts arrive.
- **Use the full sample but report honest uncertainty:** resample whole days, not pairs, because neighbouring boxes are not independent.
- **Run controls:** the shuffle test shows the skill comes from AIFS, not from the method.

Play A shows how quickly skill degrades with only a few weeks of data, which tells a service how long to wait before trusting a new model.

## lab5.intro
Early-warning systems turn forecasts into **decisions**. A probability is useful only if it is **reliable** (when it says 30%, the event happens about 30% of the time), **sharp** enough to act on, and tied to a decision rule. This lab builds from probabilities to warnings to a simple **risk index**.

## lab5.predict.q
Raw AIFS can only say "yes" or "no" to heavy rain. Calibrated AIFS gives a probability. Who gains more from the probability: a user whose protective action is **cheap** (say, moving goods to higher ground) or one whose action is **expensive** (evacuating a village)?

## lab5.predict.a
**Both**, in different ways, and Play C shows why. A yes/no forecast fixes a single trade-off between misses and false alarms. A probability lets each user set their own trigger: cheap actions at a low probability, expensive actions only at a high one. The value of a forecast depends on the user's costs, not only on its score.

## lab5.rel.explain
**Reliability diagram.** Each point groups forecasts by their stated probability and shows how often the event actually happened. On the diagonal means the probabilities can be taken at face value.

- **Calibrated AIFS** should lie close to the diagonal and use a wide range of probabilities: confident when the weather supports it.
- **Climatology** is usually reliable but rarely confident; its probabilities cluster around the base rate.
- **Raw AIFS** has only two points, "no" (0%) and "yes" (100%), and both are far from what actually happened.

The bars underneath show **sharpness**: how often each probability was used. A useful forecast is both reliable and sharp.

## lab5.rel.maths
For the event $y > T$: $\mathrm{BS} = \frac{1}{n}\sum (p_i - o_i)^2$ and $\mathrm{BSS} = 1 - \mathrm{BS}/\mathrm{BS}_{\text{clim}}$. The Brier score decomposes into reliability − resolution + uncertainty, the binary counterpart of MCB − DSC + UNC. The CRPS is the Brier score integrated over all thresholds $T$.

## lab5.roc.explain
Each warning rule ("warn when p ≥ cutoff") gives one trade-off between the **hit rate** and the **false alarm rate**. The blue curve traces every possible rule; the further it bows towards the top left, the better the forecast discriminates (AUC 0.5 is no skill, 1 is perfect). Raw AIFS offers a single point, and its trade-off may suit nobody. The ROC ignores calibration: it asks only whether events get higher probabilities than non-events.

## lab5.value.explain
Suppose protecting yourself costs **C** and an unprotected heavy-rain event costs **L**. A user with cost/loss ratio C/L acts when the probability exceeds C/L. **Relative economic value** measures how much of the gap between using climatology and having a perfect forecast this closes (1 = perfect, 0 = no better than climatology, below 0 = worse).

At C/L = **{cl}**, calibrated AIFS delivers **{v_idr}** of the possible value and raw AIFS **{v_raw}**. Notice that raw AIFS helps only users near one particular C/L, while calibrated probabilities help a much wider range of users.

## lab5.value.maths
Expected expense per unit loss: $E = \alpha\,(h + f) + m$, with $\alpha = C/L$ and $h, f, m$ the shares of hits, false alarms and misses. Then $V = (E_{\text{clim}} - E_{\text{fc}}) / (E_{\text{clim}} - E_{\text{perfect}})$, with $E_{\text{clim}} = \min(\alpha, \bar o)$ and $E_{\text{perfect}} = \alpha \bar o$ (Richardson, 2000).

## lab5.index.intro
Many African meteorological services issue **impact-based warnings** from a matrix of **likelihood × severity**. Calibrated probabilities make this matrix quantitative: the likelihood comes straight from the calibrated forecast, and severity from rain thresholds you choose. The result is a simple **heavy-rain warning index** for every grid box. A full risk index would also need **exposure and vulnerability** layers (people, crops, roads, informal settlements), which is an open challenge discussed in Lab 7.

## lab5.index.explain
On held-out 2026 days, **{caught}** of grid-box days with observed rain above {s2} mm/day received an orange or red warning. The table shows the other side: how often each warning level was followed by rain of each severity. Change the thresholds and likelihood bands to explore the trade-off. Lower bands catch more events but issue more false alarms, which erodes trust in warnings.

## lab5.check.q
Design a warning rule for rain above {T} mm/day. Which probability cutoff would you choose for a ministry that evacuates people, and which for a farmer deciding whether to delay fertiliser? Justify it with the ROC and value curves.

## lab5.check.a
The calibrated forecast here has AUC **{auc}** and BSS **{bss}**; raw AIFS yes/no has BSS **{bss_raw}**. An **expensive** action (evacuation, high C/L) calls for a **high** cutoff: fewer false alarms, accepting some misses. A **cheap** action (delaying fertiliser, low C/L) calls for a **low** cutoff: act often. Calibration lets one forecast serve both users; a single yes/no forecast forces them to share one trade-off.

## lab6.intro
Skill is not the same everywhere. National Meteorological Services need to know **where** AIFS adds value over climatology in their own region and climate, not just on average over Africa.

## lab6.predict.q
Where would you expect calibrated AIFS to beat climatology most clearly: wet tropical zones, semi-arid zones or deserts? Why?

## lab6.predict.a
Typically the **wet tropics and savannahs** gain most from calibration, because there is most rain to get wrong. **Deserts** score small CRPS values whatever the forecast, because it rarely rains, so skill there is fragile and noisy. Check the zone table: skill against climatology depends both on how much AIFS knows and on how informative climatology already is.

## lab6.maps.explain
The maps show each grid box's score over all 2026 test days. Red patches are where the forecast loses to climatology. These can come from convective rain that no model places correctly, from sparse IMERG quality, or from a zone model that fits the zone's average box but not this one. Pick a region to zoom in on a climate centre's area of responsibility.

## lab6.zones.explain
One row per climate zone, largest first. **CRPSS** against climatology near zero means AIFS adds little beyond the usual rain for that place and time of year. The **error removed by calibration** is usually large everywhere, because raw single-number forecasts are heavily penalised by the CRPS.

## lab6.check.q
You advise a regional climate centre. Using the region table for lead day {lead}, where would you trust calibrated AIFS most, and where least? What would you check before using it operationally?

## lab6.check.a
At lead day {lead}, calibrated AIFS is most skilful against climatology in **{best}** ({best_v}) and least in **{worst}** ({worst_v}). Before operational use: check skill **by season and by station** rather than area averages; compare against a **local** climatology the service already uses; verify against **gauges** where available, since IMERG is itself uncertain; and repeat the analysis as more AIFS data arrive.

## lab7.intro
The methods in this workshop work, but the problem is far from solved. These are the open questions where statistics, machine learning and climate science meet, each linked to the lab that shows it.

## lab7.q1.q
What are the three components of the decomposition, and which is most important for operational utility?

## lab7.q1.a
Miscalibration (MCB), discrimination (DSC) and uncertainty (UNC). **DSC** matters most: miscalibration can be removed by post-processing, uncertainty is fixed by the climate, and discrimination is the information only the forecast can provide. At lead day {lead}: MCB {mcb}, DSC {dsc}, UNC {unc} mm/day.

## lab7.q2.q
AIFS-Single gives one rainfall number per grid box and day, while climatology is a distribution. What terms describe these forecasts, and are they comparable in skill?

## lab7.q2.a
AIFS-Single is a **deterministic (point) forecast**; climatology is a **probabilistic (distributional) forecast**. The CRPS scores both on the same scale, since for a point forecast it reduces to the absolute error, so they can be compared. The comparison is unfair to the point forecast, which cannot express uncertainty, and that is why we calibrate it first.

## lab7.q3.q
How can we best measure AIFS-Single's discrimination against climatology?

## lab7.q3.a
Calibrate it first (IDR), then compare the **calibrated** CRPS with climatology's: $\mathrm{DSC} = \mathrm{CRPS}_{clim} - \mathrm{CRPS}_{cal}$, or $\mathrm{CRPSS}_{cal}$. For specific events, use the ROC area or the resolution term of the Brier score, which do not reward or penalise calibration.

## lab7.q4.q
Given that AIFS-Single is new, with only two years available to recalibrate, what can we do to ensure enough samples within the distribution to train on?

## lab7.q4.a
Pool grid boxes with similar climates (space for time), use a method with few assumptions such as IDR that needs no parameters, update models as data arrive, and quantify uncertainty honestly by resampling whole days. Lab 4 shows each of these.

## lab7.challenges
**1. Short records and non-stationarity.** AI weather models are retrained and upgraded often; each version may need recalibrating, and a single training year cannot represent ENSO or decadal variability. How do we calibrate a moving target? *(Lab 4A)*

**2. Noisy, uncertain truth.** IMERG is a satellite estimate, and the 2026 test data use the Late run because the Final run is not yet out. Gauge networks are thin. Skill scores change with the reference dataset, so how do we verify forecasts against an uncertain truth? *(Lab 1)*

**3. Extremes beyond the training range.** IDR does not extrapolate, and heavy-rain days are few. Hybrid methods with extreme-value tails, or physically informed pooling, are active research. *(Lab 2, Lab 5)*

**4. Deterministic vs ensemble AI.** Ensemble AI models (GenCast, FGN, AIFS-ENS) provide spread directly. Is their spread calibrated, and does IDR on the ensemble mean throw information away? *(Lab 3)*

**5. Dependence and honest uncertainty.** Grid boxes and days are correlated, so naive confidence intervals are far too narrow. Spatial and temporal block methods are essential. *(Lab 4C)*

**6. From hazard to risk.** A warning index uses hazard probabilities only. Climate risk also needs exposure and vulnerability data, which are often missing or outdated in exactly the places most at risk. *(Lab 5D)*

**7. Co-production with National Meteorological Services.** Thresholds, warning matrices and verification should be designed with forecasters and users, and the tools must run on the infrastructure services actually have.

## glossary
**AIFS-Single.** ECMWF's Artificial Intelligence Forecasting System, a machine-learning weather model. "Single" means one deterministic run rather than an ensemble.

**IMERG.** NASA's Integrated Multi-satellitE Retrievals for GPM: gridded rainfall estimated from satellites, at 0.1°. The Final run is gauge-adjusted; the Late run is faster but less accurate.

**Lead day.** Lead day 2 is the rain 24–48 hours after the 00 UTC forecast start.

**Climatology.** A forecast that ignores the weather model and uses past observations for the same place and time of year. Here: all IMERG days within ±15 days of the date in 2000–2020.

**Calibration / post-processing.** Statistically correcting a model's output so that its probabilities match observed frequencies.

**IDR (isotonic distributional regression).** A post-processing method that learns a full predictive distribution for each forecast value, assuming only that wetter forecasts mean stochastically wetter outcomes.

**PAVA (pool adjacent violators algorithm).** The algorithm IDR uses to enforce the order rule, by averaging neighbouring values that break it.

**Predictive distribution / CDF.** For each rain amount, the probability that the observed rain is at or below it.

**Köppen–Geiger zone.** A climate class (such as Aw, tropical savannah) used here to decide which grid boxes share a model.

**Space for time.** Pooling data across locations to make up for a short record at each location.

**CRPS (continuous ranked probability score).** The error of a probability forecast, in mm/day. It equals the absolute error for a single-number forecast. Lower is better.

**CRPSS.** CRPS skill score: 1 − CRPS / CRPS of climatology. 0 means no better than climatology; positive is better.

**MCB, DSC, UNC.** Miscalibration, discrimination and uncertainty: the three parts of the score decomposition CRPS = MCB − DSC + UNC.

**Brier score / BSS.** The CRPS's counterpart for a single yes/no event such as rain > 10 mm/day, and its skill score.

**Reliability diagram.** A plot of forecast probability against observed frequency. Points on the diagonal mean the probabilities can be trusted.

**Sharpness.** How far forecasts move away from climatology towards confident probabilities.

**ROC curve / AUC.** Hit rate against false alarm rate for every warning cutoff. AUC is the area under the curve; 0.5 means no skill.

**Relative economic value.** The share of the possible benefit of a perfect forecast that a real forecast delivers, for a user with a given cost/loss ratio.

**Impact-based warning matrix.** A grid of likelihood × severity that sets the warning colour.

**Block bootstrap.** Resampling whole blocks, here whole days, to estimate uncertainty when data within a block are correlated.

**Out-of-sample.** Scored on data not used for fitting, such as the 2026 test days.

## data.intro
Before any statistics, know what is being compared. This page sets out the four datasets, the periods used for training and testing, how a forecast is matched with an observation, what daily rainfall looks like, and the conventions and caveats behind every number in the labs.

## data.predict.q
Across all land grid boxes in the 2026 test period, what share of grid-box days had **less than 1 mm** of rain according to IMERG?

## data.predict.a
About **{p_all}** across all zones, and **{p_zone}** in zone {zone}. Most days are dry or nearly dry, so a forecast's distribution must put a large probability on (almost) zero rain. That is one reason to work with CDFs, which handle a point mass at zero, rather than densities.

## data.sources.explain
- **AIFS-Single** is a deterministic forecast: one number per grid box and day. Its 0.25° field is interpolated (bilinear) onto IMERG's 0.1° grid so both refer to the same box. This is **regridding, not downscaling**: interpolation adds no fine-scale detail. The calibration then corrects the forecast's distribution towards what IMERG records in 0.1° boxes.
- **IMERG** is the "truth" here, but it is an estimate built from satellites. The **Final** run is adjusted with rain gauges and arrives months later; the **Late** run is available within a day but is not gauge-adjusted.
- **The reference forecast** answers the question "what does a forecaster know without AIFS?". It uses the rain recorded on the same calendar window over 21 years.
- **Climate zones** decide which grid boxes share a calibration model.

## data.timeline.explain
Each tick is one day on which a lead-day-{lead} forecast can be matched with IMERG: **{n_train}** training days and **{n_test}** test days. The training and test periods do not overlap, so every test score is out of sample. The training set covers roughly one cycle of each region's seasons, a single version of each rainy season. The diamonds mark the heavy-rain case-study days used in Labs 5 and 6, chosen as the 2026 days with the most land area above 50 mm.

## data.lead.explain
AIFS runs start at 00 UTC, and the files store 24-hour rainfall totals ending at each 24-hour step. **Lead day 2** is the rain from 24 to 48 hours after the start, which falls on the calendar day after the run, the same 00–24 UTC day IMERG reports. Longer leads are matched to later days. Each lead day has its own calibration models, because forecast errors grow with lead time.

## data.rain.explain
Daily rainfall is a difficult variable for statistics: a large **point mass at or near zero**, a long **right tail**, and very different behaviour between climates. Three things to notice:

- **Most grid-box days are dry or nearly dry**, especially in arid zones.
- **Compare AIFS with IMERG** category by category. Weather models often spread light rain too widely and smooth out the heaviest falls; check whether AIFS does here.
- **The heaviest falls are rare**, often well under 1% of grid-box days. They are exactly the events warnings are about, and the hardest to estimate from a short record.

The table summarises each zone's observed rain in the test period, with the AIFS mean for comparison.

## data.rain.maths
Shares come from the stored random sample of held-out test pairs: up to 20,000 per zone and lead day. When all zones are pooled, each zone is weighted by its number of valid test cells, so large zones count more. Percentiles are per zone and unweighted.

## data.oneday.explain
A single day shows why point-by-point verification is hard. AIFS often places a rain system roughly right but smooths it: wider, weaker and shifted by a few grid boxes compared with IMERG's sharper, patchier field. A forecast that is close but displaced is still scored as wrong in both places, which is part of why raw AIFS loses to climatology on the CRPS.

## data.units
- **Rain amounts** are shown in **mm/day** in charts. The notebook and the score tables work in **mm/hr** (the daily mean rate), as in the original tutorial: multiply mm/hr by 24 to get mm/day.
- **Rounding:** before fitting, every forecast and observation is rounded to **0.1 mm/day**. Heavy-rain events and raw yes/no forecasts use the same rounded values, so all forecasts are judged on exactly the same event.
- **Raw AIFS is scored as a single number:** its CRPS equals the absolute error |forecast − observed|.
- **Land only:** sea grid boxes are excluded throughout.

## data.caveats
- **The truth changes between periods.** Training uses gauge-adjusted IMERG Final; the 2026 test days use the Late run. Some of the test error may come from this switch rather than from AIFS.
- **IMERG is uncertain** where gauges are few, and satellite estimates struggle with some rain types, such as warm-rain and orographic rain. Skill against IMERG is not the same as skill against rain gauges.
- **One training year** cannot show how forecast errors vary between wet and dry years or between ENSO phases.
- **The climatology period (2000–2020)** predates recent change; in a warming climate it may not represent today's rain perfectly.
- **AI weather models change.** Calibration learned for one model version may not transfer to the next.

## data.check.q
A colleague says: "The calibrated forecast beats climatology on 2026, so it will beat climatology next year too." Using this page, give two reasons to be cautious.

## data.check.a
First, the model learned from only about {n_train} days covering one cycle of seasons. Next year's rainy seasons may differ (a different ENSO phase, say), and a new AIFS version may make different errors. Second, the 2026 verification uses IMERG Late, not the Final run used in training. If the reference product changes again, so might the measured skill. Scores are estimates with uncertainty (Lab 4), conditional on the data and the period; re-verify as new forecasts and observations arrive.

## data.about.aifs
**AIFS-Single is the forecast being calibrated.** It is the deterministic version of ECMWF's **Artificial Intelligence Forecasting System**, a machine-learning weather model that runs alongside ECMWF's physics-based model (the IFS).

- **How it works.** A graph neural network encodes the state of the atmosphere, a transformer processes it, and a decoder maps it back to a global grid. The model steps forward 6 hours at a time, feeding each step back in as the next input.
- **What it learned from.** It was trained on decades of ERA5 reanalysis and fine-tuned on ECMWF's operational analyses. It learns the weather's dynamics from data rather than from solved physical equations.
- **Where it starts.** Each forecast starts from ECMWF's analysis of the current atmosphere, which still relies on conventional data assimilation.
- **Rainfall.** Rain is one of the model's output variables, on a grid of roughly 0.25° (about 28 km). We use 24-hour totals.
- **"Single" means one run.** There is no ensemble and so no built-in measure of uncertainty. That is exactly what calibration supplies.
- **A moving target.** AIFS became an operational ECMWF product in 2025 and is updated from time to time. The 2024 training forecasts predate operational status, so later versions may make somewhat different errors.

## data.about.imerg
**GPM IMERG is the "truth" the forecasts are judged against.** It is NASA's **Integrated Multi-satellitE Retrievals for the Global Precipitation Measurement mission**, here version 7, available every 30 minutes at 0.1° and summed to daily (00–24 UTC) totals.

- **How it is built.** Microwave sensors on a constellation of satellites estimate rain as they pass over, all calibrated to the radar and microwave imager on the GPM Core Observatory. Between passes, infrared images from geostationary satellites, which see cloud tops rather than rain, fill the gaps.
- **Three runs of the same product.** **Early** (about 4 hours after real time) and **Late** (about half a day after) are fast. **Final** comes roughly 3.5 months later and is adjusted to monthly rain-gauge totals. Training here uses Final; the 2026 test days use Late.
- **Known weaknesses.** Satellite estimates can miss or misjudge light rain and drizzle, warm-cloud and mountain rain, and the most intense convective downpours. Gauge adjustment helps only where gauges exist, and much of Africa has few.
- **Why use it anyway.** It is consistent, gridded and available everywhere, every day, which no gauge network in Africa can match. Its errors are part of what "noisy data" means in this workshop.

## data.about.clim
**The IMERG climatology is the reference forecast.** For each grid box and day, it is the set of IMERG daily totals for the same calendar window (±15 days) in every year from 2000 to 2020: 31 days × 21 years, about 650 values. It is used as an ensemble forecast, so its CRPS and its probability of heavy rain can be calculated directly.

It knows the place and the season, but nothing about the current weather. That is what a forecaster knows without a weather model. A forecast that cannot beat it adds no information, which is why every skill score in the labs is measured against it.

## data.about.kg
**Köppen–Geiger zones decide which grid boxes share a calibration model.** The Köppen–Geiger system classifies climate from long-term monthly temperature and rainfall.

- **First letter, main group:** A tropical, B arid, C temperate, D continental, E polar.
- **Second letter:** the rainfall pattern, such as f (no dry season), m (monsoon), w (dry winter), s (dry summer), or for arid climates W (desert) and S (steppe).
- **Third letter (some classes):** temperature, such as h (hot) or k (cold).

For example, **Aw** is tropical savannah with a dry winter, and **BWh** is hot desert. The map used here comes from Beck et al. (2023), regridded to IMERG's 0.1° grid. Pooling by zone assumes that grid boxes with similar climates also have similar forecast errors. Lab 4 tests that assumption.
