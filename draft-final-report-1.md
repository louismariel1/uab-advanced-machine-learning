# Centralized Hospital Laundry Location Study

## Vallès Occidental, Catalonia — Updated Project Report

**Report status:** Interim analytical report — location screening completed; final premises validation pending\
**Study area:** Vallès Occidental, Barcelona, Catalonia, Spain\
**Reference date:** 3 November 2026

---

# Executive Summary

This project evaluates potential locations for a centralized hospital laundry facility serving seven hospital and healthcare service points in Vallès Occidental, Catalonia.

The objective is to identify a location that balances geographical proximity to demand, accessibility, property suitability, and economic considerations.

The study uses four method groups, corresponding to the location-analysis methods considered in the course:

1. Geometric Models — Center of Gravity (COG).
2. Factor Rating Model.
3. Cost Model.
4. Graph-Based Models — Simple Median and Coverage.

The seven hospital service points have a combined demand weight of 2,515 beds. Applying the project's assumed laundry-generation factor of 3.0 kg per bed per day gives an estimated total demand of 7,545 kg/day.

The four method-group outcomes are:

| Method group                             | Selected result     | Representative location   |
| ---------------------------------------- | ------------------- | ------------------------- |
| Geometric — Center of Gravity            | Theoretical COG     | 41.544555° N, 2.064943° E |
| Factor Rating                            | IA-S2 — Can Roqueta | 41.537660° N, 2.134320° E |
| Cost Model                               | IA-R2 — Cova Solera | 41.480448° N, 2.026145° E |
| Graph-Based — Simple Median and Coverage | IA-T — Els Bellots  | 41.549829° N, 2.041354° E |

The two graph-based models select the same candidate area. They are therefore represented by one result in the four-method comparison.

An equal-weight consensus calculation using one representative point per method group gives:

**Consensus coordinate: 41.528131° N, 2.066684° E.**

The consensus coordinate is a theoretical geographical reference. It is not itself a confirmed industrial property or a fifth optimization result.

The preliminary findings suggest that the Sabadell–Terrassa industrial corridor deserves priority in the next stage. Can Roqueta is particularly relevant because it combines a strong Factor Rating result with evidence of industrial warehouse properties. One advertised 2,900 m² warehouse listing in Can Roqueta reports a rental price of €5.00/m²/month and describes the property as immediately available. A separate Can Roqueta II development listing describes new industrial units with loading facilities and availability advertised for the fourth quarter of 2026.

These listings provide useful property leads, but neither should yet be treated as the final selected laundry facility. Exact property coordinates, current availability, permitted use, water and wastewater capacity, technical suitability, and total operating costs still require verification.

**The current recommendation is therefore to prioritize the Can Roqueta–Sabadell industrial corridor for detailed property validation, while retaining Els Bellots and Cova Solera as important alternatives.**

---

# 1. Introduction

## 1.1 Problem Definition

Hospitals generate substantial quantities of laundry requiring regular collection, processing, and delivery. When several hospitals operate within the same geographical region, a centralized laundry facility may provide operational and economic advantages through shared infrastructure, consolidated processing, and coordinated distribution.

The location of such a facility is an important strategic decision. A poorly located facility may increase transportation distances, delivery times, operating costs, and logistical complexity.

The project therefore considers the location of one centralized hospital laundry facility serving seven hospital and healthcare service points in Vallès Occidental.

## 1.2 Project Objective

The main objective is to identify a suitable geographical area and, subsequently, an actual industrial property for the centralized laundry facility.

The location should balance:

- Proximity to the hospital network.
- Road accessibility and transportation efficiency.
- Availability of suitable industrial premises.
- Industrial compatibility and infrastructure.
- Property rental or acquisition costs.
- Potential operational and environmental constraints.

The final outcome should be a real, identifiable property that exists, is available, and can accommodate the proposed operation. A theoretical coordinate or industrial-area recommendation alone is not sufficient to establish the final property selection.

## 1.3 Study Area

The study covers the hospital network in Vallès Occidental, including Sabadell, Terrassa, and Sant Cugat del Vallès, with additional industrial-area alternatives in Rubí and Cerdanyola del Vallès.

The principal industrial search corridor includes:

- Terrassa.
- Sabadell.
- Rubí.
- Sant Cugat del Vallès.
- Cerdanyola del Vallès.

The selection of these areas provides a manageable set of alternatives covering the main geographical and industrial corridors surrounding the hospital network.

## 1.4 Centralized Laundry Concept

The proposed facility would operate as a centralized processing and distribution point.

Its logistics would include:

1. Collection of soiled laundry from the seven healthcare service points.
2. Transportation to the centralized facility.
3. Sorting, washing, disinfection, drying, finishing, and preparation.
4. Storage and dispatch of clean laundry.
5. Distribution back to the healthcare facilities.

The facility must therefore accommodate industrial laundry equipment, loading and unloading operations, appropriate utility infrastructure, storage, staff facilities, and vehicle circulation.

The actual required building area and technical specifications must be established during the detailed facility-design stage.

---

# 2. Data Collection and Assumptions

## 2.1 Hospital Network

The study uses seven hospital and healthcare service points.

| ID   | Hospital / Service Point                    | Municipality          | Demand Weight (Beds) |
| ---- | ------------------------------------------- | --------------------- | -------------------- |
| H01  | Hospital de Sabadell – Parc Taulí           | Sabadell              | 861                  |
| H02  | Hospital de Terrassa                        | Terrassa              | 460                  |
| H03  | Mútua de Terrassa + Àptima Centre Clínic    | Terrassa              | 604                  |
| H04  | Hospital Universitari General de Catalunya  | Sant Cugat del Vallès | 304                  |
| H05  | Centre de Prevenció i Rehabilitació Asepeyo | Sant Cugat del Vallès | 138                  |
| H06  | Hospital de Sant Llàtzer                    | Terrassa              | 77                   |
| H07  | Hospital Quirónsalud del Vallès             | Sabadell              | 71                   |
| <br> | **Total**                                   | <br>                  | **2,515**            |

The bed figures are used as relative demand weights. They do not represent measured laundry consumption at each hospital.

## 2.2 Hospital Coordinates

The geographical dataset uses WGS84 decimal-degree coordinates.

| ID  | Latitude | Longitude |
| --- | -------- | --------- |
| H01 | 41.55719 | 2.10989   |
| H02 | 41.55679 | 2.05277   |
| H03 | 41.56383 | 2.01718   |
| H04 | 41.47491 | 2.04463   |
| H05 | 41.49010 | 2.07843   |
| H06 | 41.56261 | 2.01721   |
| H07 | 41.53233 | 2.11760   |

The master geographical reference is WGS84 (EPSG:4326).

For projected spatial calculations, the study uses ETRS89 / UTM Zone 31N (EPSG:25831), which is appropriate for the study area.

The geographical coordinates represent the hospital or healthcare service-point locations rather than the precise location of a laundry loading dock.

## 2.3 Demand Estimation

The project adopts an assumed laundry-generation factor of:

[\
3.0\ \text{kg per bed per day}\
]

Total estimated daily demand is:

[\
D=2,515\times3.0\
]

[\
\boxed{D=7,545\ \text{kg/day}}\
]

This corresponds to approximately 7.545 tonnes of laundry per day.

The estimate is a project assumption used for preliminary location analysis. It is not a measurement of actual hospital laundry consumption.

The three largest service points account for:

[\
861+604+460=1,925\ \text{beds}\
]

[\
\frac{1,925}{2,515}\times100=76.5%\
]

Consequently, approximately 76.5% of the demand weight is concentrated in H01, H02, and H03, around Sabadell and Terrassa.

This concentration has a substantial influence on the geographical results.

## 2.4 Candidate Industrial Areas

Six candidate industrial areas were retained for the comparative analysis.

| ID    | Candidate Area                                | Municipality / Corridor |
| ----- | --------------------------------------------- | ----------------------- |
| IA-T  | Els Bellots / Santa Margarida / Can Parellada | Terrassa                |
| IA-S1 | Sud-Oest / Sant Pau de Riu-sec                | Sabadell                |
| IA-S2 | Sabadell Parc Empresarial / Can Roqueta       | Sabadell                |
| IA-R1 | Can Sant Joan / Can Jardí                     | Rubí                    |
| IA-R2 | Central and southern industrial corridor      | Rubí                    |
| IA-C  | Cerdanyola industrial corridor                | Cerdanyola del Vallès   |

These areas were selected to represent the principal industrial alternatives around the hospital network.

They are analytical candidate areas, not individual properties.

The selection is supported by the industrial geography of the municipalities. Sabadell's industrial areas include Can Roqueta, Sud-Oest, Sabadell Parc Empresarial, and other established activity zones. Terrassa and Rubí also contain extensive industrial corridors.

## 2.5 Candidate Representative Coordinates

A representative point was selected for each candidate area to support preliminary geographical comparison.

| Candidate | Representative Area | Latitude  | Longitude |
| --------- | ------------------- | --------- | --------- |
| IA-T      | Els Bellots         | 41.549829 | 2.041354  |
| IA-S1     | Sant Pau de Riu-sec | 41.513398 | 2.099462  |
| IA-S2     | Can Roqueta         | 41.537660 | 2.134320  |
| IA-R1     | Can Jardí           | 41.473458 | 2.020948  |
| IA-R2     | Cova Solera         | 41.480448 | 2.026145  |
| IA-C      | Parc de l'Alba      | 41.485460 | 2.107650  |

These coordinates are representative analytical points. They are not confirmed building entrances, parcel centroids, or available property coordinates.

## 2.6 Main Assumptions

| Parameter                            | Adopted Value / Convention    |
| ------------------------------------ | ----------------------------- |
| Hospital demand weight               | Number of beds                |
| Laundry-generation factor            | 3.0 kg/bed/day                |
| Estimated total demand               | 7,545 kg/day                  |
| Master coordinate system             | WGS84, EPSG:4326              |
| Projected coordinate system          | ETRS89 / UTM 31N, EPSG:25831  |
| Number of hospital service points    | 7                             |
| Number of candidate industrial areas | 6                             |
| Coverage baseline radius             | 7.5 km                        |
| Factor Rating scale                  | 1–5                           |
| Method consensus                     | Equal weight per method group |
| Property cost comparison             | Asking rent per m²/month      |

## 2.7 Important Data Limitations

The current analysis uses different levels of geographical and property detail.

In particular:

- Hospital coordinates are service-point references.
- Candidate industrial-area coordinates are representative points.
- Property listing information does not always disclose an exact street address or geographical coordinate.
- Asking rents are not equivalent to total facility operating costs.
- The current graph-model distance calculations use straight-line geographical distances, not validated road-network distances.

These distinctions are maintained throughout the results.

---

# 3. Geometric Model — Center of Gravity

## 3.1 Method

The Center of Gravity method identifies a theoretical location based on the geographical coordinates and demand weights of the hospital service points.

The method seeks to locate a point that minimizes the weighted sum of squared Euclidean distances to the demand locations.

For projected coordinates:

[\
X\_{COG}=\frac{\sum\_{i=1}^{n}W_iX_i}{\sum\_{i=1}^{n}W_i}\
]

[\
Y\_{COG}=\frac{\sum\_{i=1}^{n}W_iY_i}{\sum\_{i=1}^{n}W_i}\
]

where:

- (X_i) = projected easting of hospital (i);
- (Y_i) = projected northing of hospital (i);
- (W_i) = demand weight of hospital (i);
- (n) = number of hospital service points.

The calculation uses projected metric coordinates rather than directly averaging latitude and longitude degrees.

## 3.2 Input Data

The calculation uses all seven hospital service points and their corresponding bed weights.

Total demand weight:

[\
\sum W_i=2,515\
]

Coordinate system:

[\
\text{ETRS89 / UTM Zone 31N (EPSG:25831)}\
]

## 3.3 Calculation Result

The projected Center of Gravity is:

[\
E=422,010.609\ \text{m}\
]

[\
N=4,599,632.802\ \text{m}\
]

Converted to WGS84:

[\
\boxed{41.544555^\circ N,\quad2.064943^\circ E}\
]

This is the theoretical geographical Center of Gravity of the seven hospital service points, weighted by their respective bed numbers.

## 3.4 Interpretation

The result lies within the broader Sabadell–Terrassa geographical corridor.

The location is strongly influenced by the concentration of demand around H01, H02, and H03.

The Center of Gravity provides a useful geographical reference for subsequent industrial-area screening.

However, it does not account directly for:

- Road-network geometry.
- Traffic conditions.
- Actual travel times.
- Industrial zoning.
- Building availability.
- Facility size and infrastructure.
- Property prices.

It should therefore not be interpreted as the final operational facility location.

## 3.5 Geographical Identification

The COG coordinate can be inspected using a reverse-geocoding service or a mapping application.

The coordinate itself has been established mathematically. Its exact postal address has not yet been confirmed through a successful reverse-geocoding result.

Map reference:

[https://www.google.com/maps?q=41.544555,2.064943](https://www.google.com/maps?q=41.544555,2.064943)

---

# 4. Factor Rating Model

## 4.1 Method

The Factor Rating Model evaluates candidate locations using several qualitative and operational criteria.

Each criterion receives a weight representing its relative importance. Each candidate is then assigned a score from 1 to 5.

The weighted score is:

[\
S_j=\sum\_{k=1}^{m}w_ks\_{jk}\
]

where:

- (S_j) = weighted score of candidate (j);
- (w_k) = normalized weight of criterion (k);
- (s\_{jk}) = score of candidate (j) for criterion (k);
- (m) = number of evaluation criteria.

The maximum possible score is 5.

## 4.2 Evaluation Criteria and Weights

| Criterion                                   | Weight   |
| ------------------------------------------- | -------- |
| Proximity to hospital demand                | 25%      |
| Road accessibility                          | 20%      |
| Evidence of suitable premises               | 20%      |
| Industrial compatibility                    | 15%      |
| Utilities and infrastructure                | 10%      |
| Environmental and residential compatibility | 10%      |
| **Total**                                   | **100%** |

The criteria reflect the principal geographical, logistical, property, and operational considerations for a centralized industrial laundry.

## 4.3 Candidate Scores

Scores are preliminary screening judgments based on the available geographical and property evidence.

| Candidate             | Proximity | Roads | Premises | Industrial | Utilities | Environment |
| --------------------- | --------- | ----- | -------- | ---------- | --------- | ----------- |
| IA-T — Els Bellots    | 5         | 4     | 3        | 5          | 3         | 3           |
| IA-S1 — Sant Pau      | 4         | 4     | 2        | 5          | 3         | 3           |
| IA-S2 — Can Roqueta   | 3         | 5     | 5        | 5          | 4         | 3           |
| IA-R1 — Can Jardí     | 2         | 4     | 4        | 5          | 4         | 3           |
| IA-R2 — Cova Solera   | 2         | 4     | 4        | 5          | 4         | 3           |
| IA-C — Parc de l'Alba | 2         | 5     | 2        | 3          | 4         | 3           |

## 4.4 Weighted Results

| Rank | Candidate                   | Weighted Score |
| ---- | --------------------------- | -------------- |
| 1    | **IA-S2 — Can Roqueta**     | **4.15 / 5**   |
| 2    | IA-T — Els Bellots          | 4.05 / 5       |
| 3    | IA-S1 — Sant Pau de Riu-sec | 3.65 / 5       |
| 4    | IA-R1 — Can Jardí           | 3.55 / 5       |
| 4    | IA-R2 — Cova Solera         | 3.55 / 5       |
| 6    | IA-C — Parc de l'Alba       | 3.00 / 5       |

The highest preliminary score is obtained by:

[\
\boxed{\text{IA-S2 — Can Roqueta}}\
]

with a score of 4.15/5.

## 4.5 Interpretation

Can Roqueta obtains the highest score because it combines:

- Strong industrial compatibility.
- Good road accessibility.
- Evidence of industrial warehouse premises.
- Available property information.
- A relatively strong infrastructure profile.

Els Bellots obtains a very similar score of 4.05/5.

The difference between the two candidates is only 0.10 points. Consequently, the ranking should be treated as indicative rather than conclusive.

The Factor Rating result is also sensitive to the quality of the underlying evidence. In particular, utilities and environmental compatibility have not yet been validated through property-specific technical documentation.

The result should therefore be refined when the actual premises are shortlisted.

---

# 5. Cost Model

## 5.1 Method and Scope

The Cost Model is intended to compare the economic consequences of selecting different industrial areas.

A complete cost model should consider:

[\
C\_{Total}=C\_{Facility}+C\_{Transport}+C\_{Utilities}+C\_{Labour}+C\_{Equipment}+C\_{Other}\
]

The current project data do not provide sufficiently comparable information for all these cost categories.

Consequently, the present calculation is a **preliminary facility-rental-cost comparison**, not a complete total-cost-of-operation model.

## 5.2 Available Rental Evidence

The following asking-rent information was identified for the candidate areas.

| Candidate              | Listed Area                      | Asking Rent    |
| ---------------------- | -------------------------------- | -------------- |
| IA-T — Els Bellots     | 185 m²                           | €5.95/m²/month |
| IA-S2 — Can Roqueta    | 2,900 m²                         | €5.00/m²/month |
| IA-R1 — Can Jardí      | 1,001 m²                         | €4.80/m²/month |
| IA-R2 — Cova Solera    | 1,500 m²                         | €4.00/m²/month |
| IA-C — Parc Tecnològic | 3,300 m²                         | €5.45/m²/month |
| IA-S1 — Sant Pau       | No comparable listing identified | N/A            |

These are asking-rent observations from property listings. They are not confirmed contractual rents or verified long-term costs.

## 5.3 Normalized Rental Comparison

To compare the different locations on a common basis, the asking rents were normalized to a hypothetical 1,000 m² facility.

[\
C\_{Annual}=R\times1,000\times12\
]

where (R) is the asking rent per square metre per month.

| Candidate              | Rent €/m²/month | Illustrative Annual Rent for 1,000 m² |
| ---------------------- | --------------- | ------------------------------------- |
| IA-R2 — Cova Solera    | 4.00            | €48,000                               |
| IA-R1 — Can Jardí      | 4.80            | €57,600                               |
| IA-S2 — Can Roqueta    | 5.00            | €60,000                               |
| IA-C — Parc Tecnològic | 5.45            | €65,400                               |
| IA-T — Els Bellots     | 5.95            | €71,400                               |
| IA-S1 — Sant Pau       | N/A             | N/A                                   |

## 5.4 Cost Model Result

The lowest observed asking rent is associated with:

[\
\boxed{\text{IA-R2 — Cova Solera}}\
]

at approximately €4.00/m²/month.

This gives an illustrative annual rent of €48,000 for a hypothetical 1,000 m² facility.

## 5.5 Interpretation and Limitations

Cova Solera is the lowest-cost candidate based on the available asking-rent evidence.

However, this does not establish that Cova Solera offers the lowest total operating cost.

The current comparison does not include:

- Actual laundry transport distances and travel times.
- Fuel, vehicle, and fleet costs.
- Water consumption.
- Wastewater treatment and discharge costs.
- Electricity and gas consumption.
- Building adaptation and installation costs.
- Laundry equipment and maintenance.
- Labour costs.
- Taxes, service charges, insurance, and other property costs.

Furthermore, the properties differ in size and characteristics, and the listings may change or cease to be available.

The result must therefore be interpreted as a rental-cost screening result only.

---

# 6. Graph-Based Models

Two graph-related models are included in this method group:

1. Simple Median Model.
2. Coverage Model.

Both models were evaluated using the seven hospital service points and the six candidate industrial areas.

**Important methodological qualification:** the current numerical screening used straight-line geographical distances between representative points. A fully implemented graph-based model should instead use a road-network graph and shortest-path distances. The results below are therefore preliminary geographical proxies, not final road-network optimization results.

## 6.1 Simple Median Model

### 6.1.1 Method

The Simple Median Model seeks the candidate location that minimizes the total weighted distance to the hospital service points.

For candidate (j):

[\
D_j=\sum\_{i=1}^{n}W_id\_{ij}\
]

where:

- (W_i) = demand weight of hospital (i);
- (d\_{ij}) = distance between hospital (i) and candidate (j).

For comparison, the results are presented as weighted mean distance:

[\
\bar d_j=\frac{\sum_iW_id\_{ij}}{\sum_iW_i}\
]

The current screening uses geodesic straight-line distances.

### 6.1.2 Results

| Rank | Candidate                   | Weighted Mean Distance |
| ---- | --------------------------- | ---------------------- |
| 1    | **IA-T — Els Bellots**      | **4.48 km**            |
| 2    | IA-S1 — Sant Pau de Riu-sec | 6.22 km                |
| 3    | IA-S2 — Can Roqueta         | 6.74 km                |
| 4    | IA-C — Parc de l'Alba       | 8.46 km                |
| 5    | IA-R2 — Cova Solera         | 8.61 km                |
| 6    | IA-R1 — Can Jardí           | 9.37 km                |

### 6.1.3 Result

The best candidate under the current weighted-distance screening is:

[\
\boxed{\text{IA-T — Els Bellots}}\
]

The candidate has the lowest weighted mean straight-line distance to the hospital network.

### 6.1.4 Interpretation

Els Bellots benefits from its position between the major hospital demand concentrations.

It is particularly close to the Terrassa and Sabadell demand centres.

However, road-network distances and travel times may change the ranking. The result should therefore be recalculated using a road-network graph before it is treated as the definitive Simple Median result.

## 6.2 Coverage Model

### 6.2.1 Method

The Coverage Model evaluates how much hospital demand is located within a specified distance of each candidate facility.

For candidate (j) and coverage radius (R):

[\
C_j(R)=\sum\_{i=1}^{n}W_iI(d\_{ij}\leq R)\
]

where:

- (W_i) = hospital demand weight;
- (d\_{ij}) = distance from hospital (i) to candidate (j);
- (I(\cdot)) = indicator function, equal to 1 when the distance is within the coverage radius and 0 otherwise.

The baseline coverage radius is:

[\
\boxed{R=7.5\text{ km}}\
]

This is a project modelling assumption, not a statutory or operational service requirement.

### 6.2.2 Baseline Coverage Results

| Rank | Candidate                   | Covered Beds | Coverage  |
| ---- | --------------------------- | ------------ | --------- |
| 1    | **IA-T — Els Bellots**      | **2,211**    | **87.9%** |
| 2    | IA-S1 — Sant Pau de Riu-sec | 1,834        | 72.9%     |
| 3    | IA-S2 — Can Roqueta         | 1,530        | 60.8%     |
| 4    | IA-C — Parc de l'Alba       | 513          | 20.4%     |
| 5    | IA-R1 — Can Jardí           | 442          | 17.6%     |
| 5    | IA-R2 — Cova Solera         | 442          | 17.6%     |

### 6.2.3 Sensitivity to Coverage Radius

The coverage calculation was repeated using three distance thresholds.

| Candidate             | 5 km  | 7.5 km | 10 km |
| --------------------- | ----- | ------ | ----- |
| IA-T — Els Bellots    | 45.4% | 87.9%  | 100%  |
| IA-S1 — Sant Pau      | 42.5% | 72.9%  | 100%  |
| IA-S2 — Can Roqueta   | 37.1% | 60.8%  | 60.8% |
| IA-R1 — Can Jardí     | 12.1% | 17.6%  | 38.9% |
| IA-R2 — Cova Solera   | 17.6% | 17.6%  | 65.8% |
| IA-C — Parc de l'Alba | 5.5%  | 20.4%  | 72.9% |

### 6.2.4 Result

Els Bellots obtains the highest baseline coverage:

[\
\boxed{87.9%\text{ of weighted hospital demand}}\
]

The result remains strong across the three tested coverage radii.

### 6.2.5 Interpretation

Els Bellots offers the strongest geographical coverage of the six candidates under the present straight-line screening.

The result reinforces its position as a strong candidate for further investigation.

However, a final coverage analysis should use actual road-network distances or travel times, particularly because hospital laundry collection and delivery are operational transport activities.

## 6.3 Overall Graph-Based Result

Both preliminary models select the same candidate:

[\
\boxed{\text{IA-T — Els Bellots}}\
]

This agreement provides useful supporting evidence, although the two models use the same underlying geographical distance approximation.

A road-network implementation remains necessary before claiming a fully validated graph-based optimum.

---

# 7. Comparison of Results

## 7.1 Comparison of the Four Method Groups

The project uses four method groups and five individual model outputs.

| Method Group  | Individual Model       | Selected Result     |
| ------------- | ---------------------- | ------------------- |
| Geometric     | Center of Gravity      | Theoretical COG     |
| Factor Rating | Weighted factor score  | IA-S2 — Can Roqueta |
| Cost Model    | Rental-cost comparison | IA-R2 — Cova Solera |
| Graph-Based   | Simple Median          | IA-T — Els Bellots  |
| Graph-Based   | Coverage               | IA-T — Els Bellots  |

The two graph models agree and therefore contribute one representative result to the four-group consensus calculation.

## 7.2 Agreement and Disagreement

The methods identify different candidate locations because they optimize different objectives.

- The Center of Gravity identifies a theoretical weighted geographical balance point.
- Factor Rating favours Can Roqueta because of its combined industrial, accessibility, and property-evidence profile.
- The Cost Model favours Cova Solera because it has the lowest observed asking rent per square metre.
- The Simple Median and Coverage screening methods favour Els Bellots because of its geographical relationship with the hospital demand network.

The disagreement is not necessarily a contradiction. It reflects the different priorities and objective functions of the methods.

## 7.3 Interpretation of the Results

The findings can be summarized as follows:

| Candidate           | Main Supporting Evidence                                            |
| ------------------- | ------------------------------------------------------------------- |
| Els Bellots         | Best preliminary weighted-distance and coverage results             |
| Can Roqueta         | Highest Factor Rating score and concrete warehouse listing evidence |
| Cova Solera         | Lowest observed asking rent per m²                                  |
| Sant Pau de Riu-sec | Second-best preliminary graph-based results                         |

Els Bellots and Can Roqueta are particularly important because they perform strongly under different methods.

Cova Solera remains an important economic alternative.

## 7.4 Consensus Centre of Gravity

To derive a single geographical reference from the four method groups, one representative coordinate was used for each group.

The four input points are:

| Method Group  | Selected Point | Latitude  | Longitude |
| ------------- | -------------- | --------- | --------- |
| Geometric     | COG            | 41.544555 | 2.064943  |
| Factor Rating | Can Roqueta    | 41.537660 | 2.134320  |
| Cost Model    | Cova Solera    | 41.480448 | 2.026145  |
| Graph-Based   | Els Bellots    | 41.549829 | 2.041354  |

The consensus calculation uses the projected ETRS89 / UTM Zone 31N coordinates.

The consensus easting and northing are calculated as the arithmetic means of the four method-group coordinates:

[\
E\_{Consensus}=\frac{E_1+E_2+E_3+E_4}{4}\
]

[\
N\_{Consensus}=\frac{N_1+N_2+N_3+N_4}{4}\
]

The resulting projected coordinate is approximately:

[\
E=422,136.16\text{ m}\
]

[\
N=4,597,807.77\text{ m}\
]

Converted to WGS84:

[\
\boxed{41.528131^\circ N,\quad2.066684^\circ E}\
]

### Interpretation

The consensus coordinate provides a central geographical reference that incorporates the results of the four method groups equally.

It is not a fifth optimization model and does not establish the best operational location by itself.

In particular, the four methods do not have identical data quality or analytical maturity. The consensus is therefore a transparent decision-support device rather than a statistically validated optimum.

Map reference:

[https://www.google.com/maps?q=41.52813085,2.06668408](https://www.google.com/maps?q=41.52813085,2.06668408)

---

# 8. Selection and Validation of a Real Premise

## 8.1 Objective

The purpose of this stage is to move from the theoretical and candidate-area results to an identifiable industrial building that can accommodate the centralized laundry facility.

The property must be evaluated for:

- Exact address and coordinates.
- Building and plot area.
- Current availability.
- Rental or purchase cost.
- Industrial land-use compatibility.
- Road and vehicle accessibility.
- Water supply and wastewater capacity.
- Electrical and thermal energy capacity.
- Loading and unloading facilities.
- Fire safety and building compliance.
- Potential for laundry equipment installation and internal logistics.

## 8.2 Consensus Coordinate and Nearby Industrial Search

The consensus coordinate is:

[\
41.528131^\circ N,\quad2.066684^\circ E\
]

The nearby industrial geography includes areas that warrant further investigation.

One mapped reference point in the Can Canals industrial area of Sabadell lies approximately 2.18 km from the consensus coordinate in a straight line.

This is a geographical screening reference only. It does not identify a confirmed available laundry building.

The next property-search stage should therefore identify specific industrial buildings, record their addresses and coordinates, and calculate their distances from the consensus point.

## 8.3 Can Roqueta — Identified Property Leads

Can Roqueta is a particularly relevant property-search area because it was selected by the Factor Rating Model and has concrete industrial property evidence.

Two property leads were identified.

### Property Lead A — Industrial/Logistics Warehouse, Can Roqueta

A public listing describes an industrial/logistics warehouse in the Can Roqueta industrial estate with the following characteristics:

| Characteristic          | Advertised Information                                                       |
| ----------------------- | ---------------------------------------------------------------------------- |
| Location                | Can Roqueta, Sabadell                                                        |
| Surface area            | 2,900 m²                                                                     |
| Asking rent             | €5.00/m²/month                                                               |
| Advertised availability | Immediate                                                                    |
| Loading docks           | 2                                                                            |
| TIR access doors        | 4                                                                            |
| Office area             | Approximately 400 m²                                                         |
| External yard           | Approximately 2,500 m²                                                       |
| Utilities and services  | Water, electrical installation, fire protection, and other listed facilities |
| Road access             | C-58 and AP-7                                                                |

The advertised monthly rent is:

[\
2,900\times5.00=€14,500\
]

The illustrative annual rent is:

[\
€14,500\times12=€174,000\
]

This is a promising property lead because its advertised area, loading infrastructure, and industrial characteristics may be relevant to a centralized laundry operation.

However, the listing does not establish that the building is technically suitable for the proposed laundry process. Its exact street address and coordinates must also be obtained before it can be incorporated into the final transportation analysis.

### Property Lead B — New Industrial Units, Can Roqueta II

A second listing describes a new industrial development in Can Roqueta II.

The advertised characteristics include:

| Characteristic          | Advertised Information                                      |
| ----------------------- | ----------------------------------------------------------- |
| Individual unit area    | Approximately 1,780 m²                                      |
| Asking rent             | €12,818/month for the advertised unit                       |
| Building type           | New industrial warehouse                                    |
| Loading facilities      | 4 loading docks and 4 TIR doors for the overall development |
| Clear height            | More than 11 m                                              |
| Structure               | Concrete, without intermediate columns                      |
| Additional facilities   | Offices, changing rooms, and private yard                   |
| Energy features         | Photovoltaic panels and energy-efficiency measures          |
| Advertised availability | Fourth quarter of 2026                                      |

The advertised rent corresponds to approximately:

[\
\frac{12,818}{1,780.25}\
\approx €7.20/m^2/\text{month}\
]

The listing also indicates that the four units may be rented separately or combined, subject to the necessary building and technical modifications.

The advertised availability is the fourth quarter of 2026. It must therefore be checked against the current property status before any commitment is made.

This development may provide a modern alternative to the existing warehouse listing.

## 8.4 Preliminary Property Assessment

The Can Roqueta property leads demonstrate that the candidate area has industrial premises worth investigating.

However, neither listing should yet be designated the final selected property.

The following checks remain necessary:

| Validation Item                        | Current Status                            |
| -------------------------------------- | ----------------------------------------- |
| Industrial building exists             | Supported by property listings            |
| Candidate industrial area              | Identified                                |
| Indicative floor area                  | Available                                 |
| Asking rental cost                     | Available                                 |
| Loading infrastructure                 | Partially documented                      |
| Exact street address                   | Not yet confirmed for both property leads |
| Exact building coordinates             | Not yet confirmed                         |
| Current availability                   | Requires direct revalidation              |
| Water supply capacity                  | Not verified                              |
| Wastewater capacity                    | Not verified                              |
| Electrical and thermal capacity        | Not verified                              |
| Laundry-specific technical suitability | Not verified                              |
| Planning and permitted activity        | Not verified                              |
| Final total operating cost             | Not calculated                            |

## 8.5 Property Selection Decision

The current evidence supports prioritizing the following areas:

1. Can Roqueta — strongest combination of Factor Rating result and property evidence.
2. Els Bellots — strongest preliminary geographical results.
3. Cova Solera — lowest observed rental-cost proxy.

The final building should be selected only after comparing actual premises against the operational requirements of the laundry facility.

A property should not be selected solely because it is closest to the consensus coordinate or has the lowest asking rent.

---

# 9. Final Recommendation

## 9.1 Recommended Location Strategy

The current results support prioritizing the **Sabadell–Terrassa industrial corridor**, with particular attention to Can Roqueta and Els Bellots.

The recommendation is based on the combined evidence from the four method groups:

- Can Roqueta ranks first in the Factor Rating Model.
- Els Bellots ranks first in the preliminary Simple Median and Coverage models.
- Cova Solera offers the lowest observed rental cost.
- The theoretical Center of Gravity lies within the broader Sabadell–Terrassa corridor.

The consensus coordinate provides an additional geographical reference for the next stage of property screening.

## 9.2 Preferred Candidate Area for Property Investigation

**Can Roqueta, Sabadell, is recommended as the first area for detailed property investigation.**

The reasons are:

1. It has the highest preliminary Factor Rating score.
2. It has strong industrial and road-accessibility characteristics.
3. Public listings provide evidence of industrial warehouse premises.
4. A 2,900 m² warehouse listing offers a potentially relevant scale and loading configuration.
5. A separate new-build industrial development provides another possible property option.

This is a recommendation to prioritize the area for investigation, not a final decision to lease a particular building.

## 9.3 Alternative Locations

Els Bellots should remain a major alternative because of its strong geographical performance.

Cova Solera should remain an economic alternative because of its lower observed asking rent.

The final decision should compare the actual properties, rather than only the representative coordinates of their industrial areas.

## 9.4 Principal Risks and Limitations

The principal limitations of the current study are:

- The demand estimate is based on beds and an assumed laundry-generation factor.
- The candidate coordinates represent industrial areas rather than actual premises.
- The Factor Rating scores are provisional.
- The Cost Model currently compares asking rents only.
- The graph-based results use straight-line distances rather than road-network shortest paths.
- The consensus coordinate is an equal-weight mean of method-group results, not a formal optimization model.
- Exact property addresses, coordinates, and current availability remain to be verified.
- Property-specific utility and technical suitability have not been established.

These limitations must be addressed before making a final investment or leasing recommendation.

---

# 10. Conclusion

This study has evaluated the geographical and economic location of a centralized hospital laundry facility serving seven healthcare service points in Vallès Occidental.

The demand dataset contains 2,515 beds and produces an estimated laundry demand of 7,545 kg/day under the adopted assumption of 3.0 kg per bed per day.

Four location-analysis method groups were applied:

1. Center of Gravity.
2. Factor Rating.
3. Cost Model.
4. Graph-Based Models, comprising Simple Median and Coverage.

The methods produced different results because they evaluate different aspects of the location problem.

The Center of Gravity identified a theoretical location at 41.544555° N, 2.064943° E.

The Factor Rating Model selected Can Roqueta with a preliminary score of 4.15/5.

The rental-cost comparison selected Cova Solera, with the lowest observed asking rent of €4.00/m²/month.

The preliminary Simple Median and Coverage calculations both selected Els Bellots.

Combining one result per method group produced a consensus coordinate of:

[\
\boxed{41.528131^\circ N,\quad2.066684^\circ E}\
]

The comparison indicates that the Sabadell–Terrassa industrial corridor is the most relevant geographical area for further investigation.

Can Roqueta is the recommended first area for property validation because it combines a strong Factor Rating result with concrete industrial warehouse listing evidence.

Nevertheless, the project has not yet established a final property. The next stage must confirm the exact building coordinates and postal address, validate current availability, and assess the technical and operational suitability of the premises for industrial hospital laundry processing.

**The present conclusion is therefore a preferred industrial-area recommendation, not a final property-selection decision.**

---

# References and Evidence Record

The following public sources were used or identified during the project research. Property listings should be rechecked because asking prices and availability can change.

1. Institut Cartogràfic i Geològic de Catalunya (ICGC), Geocoding and reverse-geocoding services: [https://eines.icgc.cat/geocodificador/](https://eines.icgc.cat/geocodificador/)
2. Ajuntament de Sabadell / URBACT, *Integrated Action Plan — Green Transition in Industry, City of Sabadell*, 2026.
3. Invertica, industrial/logistics warehouse listing in Can Roqueta, Sabadell, reference BCN-IND-00247: [https://www.invertica.es/index.php/inmuebles/nave-industrial-logistica-en-sabadell-bcn-ind-00247](https://www.invertica.es/index.php/inmuebles/nave-industrial-logistica-en-sabadell-bcn-ind-00247)
4. UBICAT, *Sabadell: Proyecto de 4 naves, tipo B, con todos los servicios*, reference 1178: [https://www.ubicat.cat/es/sabadell/p_i_can_roqueta_ii/naves_industriales/ref-1178](https://www.ubicat.cat/es/sabadell/p_i_can_roqueta_ii/naves_industriales/ref-1178)
5. Public industrial property listings for Els Bellots, Can Jardí, Cova Solera, and Parc Tecnològic, as recorded in the project's comparative rental dataset.
6. Public geographical mapping references used to establish hospital and candidate-area coordinates, as recorded in the project's geographical dataset.

**End of report.**

## What has changed in this updated version

- It brings the seven-hospital dataset, demand estimate, six candidate areas, all five model outputs, and the four-group consensus into one report.
- It includes the new Can Roqueta property leads in the real-premises section.
- It distinguishes the theoretical COG, the four-method consensus, and the actual building-selection stage.
- It corrects the status of the graph models: they are preliminary straight-line-distance approximations until a road-network calculation is completed.
- It avoids presenting the rental comparison as a complete operating-cost model.

One important next deliverable: obtain the exact address and coordinates of the Can Roqueta warehouse listing and compare it with the nearest identifiable industrial buildings around the consensus point. That will let us replace the representative-area coordinates with real building locations and move toward the final property-selection stage.&#x20;
