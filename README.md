Overview
This assignment analyses the Swiss dataset (Swiss_data_set.csv) — a classic socioeconomic dataset covering 47 Swiss/French municipalities. The goal is to apply descriptive and inferential statistics to understand how factors like agriculture, education, examination performance, Catholic beliefs, and infant mortality relate to fertility rates.

Dataset
FeatureDetailsFileSwiss_data_set.csvRows47 (one per municipality/region)Columns7Target VariableFertility
Variables:

Unnamed: 0 — Region name (categorical)
Fertility — Fertility rate (continuous)
Agriculture — % males in agriculture (continuous)
Examination — % draftees with highest army exam mark (discrete)
Education — % draftees with highest education (discrete)
Catholic — % of Catholic population (continuous)
Infant.Mortality — Infant mortality per 1,000 live births (continuous)


Key Findings
Data Cleaning

No missing values found in the dataset.
Outliers identified and removed using the IQR method:

Fertility: 2 outliers removed
Education: 5 outliers removed
Infant Mortality: 1 outlier removed
Agriculture, Examination, Catholic: No outliers found



Descriptive Statistics

Fertility is slightly right-skewed; most values cluster around 70.
Agriculture is approximately normal, peaking near 60%.
Examination is right-skewed; most values fall between 10–15.
Education is right-skewed; most regions have education levels below 10.
Catholic shows a bimodal distribution — regions are either predominantly Catholic (80–100%) or predominantly non-Catholic (0–20%).
Infant Mortality is approximately normal, centered around 20.

Normalization

Min-Max normalization applied to all numeric columns in the cleaned dataset to bring all features to a [0, 1] scale for comparability.

Relationships with Fertility Rate
IndicatorRelationship with FertilityAgriculturePositive — Higher agricultural involvement → Higher fertilityEducationNegative — Higher education → Lower fertilityExaminationNegative — Higher exam performance → Lower fertilityCatholicPositive — Higher Catholic % → Higher fertilityInfant MortalityPositive — Higher infant mortality → Higher fertility

Conclusion
Both descriptive and inferential statistics provide complementary lenses for understanding demographic patterns. The analysis confirms that education and healthcare improvements are associated with lower fertility, while agricultural societies and religious traditions tend to correlate with higher fertility. These findings support evidence-based policy design for demographic challenges.
