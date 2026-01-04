"""
Small copy/paste-able evaluation samples for the chat UI.

These come from the validation split used in `src/finetune.py`:
  dataset_id = "morgan/docvqa-nanochat"
  split = "validation"

Usage:
  - Copy `EVAL_SAMPLES["docvqa_val_0"]["user"]` into the chat UI input box.
  - The `expected` field is the ground-truth answer from the dataset.
"""

EVAL_SAMPLES: dict[str, dict[str, str]] = {
    "docvqa_val_0": {
        "expected": "SARGASSO SEA TEMPERATURE",
        "user": """Document:
Page 4
Unsettled Science
Knowing that weather forecasts are reliable for a
Moreover, computer models relied upon
few days at best, we should recognize the enor-
by climate scientists predict that lower atmos-
mous challenge facing scientists seeking to pre-
pheric temperatures will rise as fast as or faster
dict climate change and its impact over the next
than temperatures at the surface. However, only
century. In spite of everyone's desire for clear
within the last 20 years have reliable global
answers, it is not surprising that fundamental
measurements of temperatures in the lower at-
gaps in knowledge leave scientists unable to
mosphere been available through the use of
make reliable predictions about future changes.
satellite technology. These measurements show
A recent report from the National Re-
little if any warming.
search Council (NRC) raises important issues,
Even less is known about the potential
including these still-unanswered questions:
positive or negative impacts of climate change.
(1) Has human activity al-
In fact, many academic
ready begun to change
Sargasso Sea Temperature
studies and field experi-
temperature and the cli-
78-
ments have demonstrated
mate, and (2) How signifi-
77
Medieval
that increased levels of car-
cant will future change be?
varm period
76 -
Little
bon dioxide can
The NRC report con-
ice age
crop and forest growth.
firms that Earth's surface
So, while some argue
temperature has risen by
that the science debate is
about 1 degree Fahrenheit
73
settled and governments
over the past 150 years.
should focus only on near-
Some use this result to
72
71 .
OF
term policies-that is empty
claim that humans are
rhetoric. Inevitably, future
causing global warming,
70 -
scientific research will help
and they point to storms or
1000
500
0
- B.C. A.D. -
500 1000 1500 2000
us understand how human
floods to say that danger-
actions and natural climate
ous impacts are already
Source: Science (1996)
change may affect the world
under way. Yet scientists remain unable to con-
and will help determine what actions may be de-
firm either contention.
sirable to address the long-term.
Geological evidence indicates that climate
Science has given us enough information
and greenhouse gas levels experience significant
to know that climate changes may pose long-
natural variability for reasons having nothing to
term risks. Natural variability and human activity
do with human activity. Historical records and
may lead to climate change that could be signif-
current scientific evidence show that Europe and
North America experienced a medieval warm
cant and perhaps both positive and negative.
Consequently, people, companies and govern-
period one thousand years ago, followed cen-
ments should take responsible actions now to
turies later by a little ice age. The geological
address the issue.
record shows even larger changes throughout
One essential step is to encourage devel-
Earth's history. Against this backdrop of large,
opment of lower-emission technologies to meet
poorly understood natural variability, it is impos-
our future needs for energy. We'll next look at
sible for scientists to attribute the recent small
the promise of technology and what is being
surface temperature increase to human causes.
done today.
ExonMobil'
www.exxon.mobil.com
2000 Exxon Mobil Corporation
Source: https://www.industrydocuments.ucsf.edu/docs/ttw10228

Question: What is the title of the chart?""",
    },
    "docvqa_val_1": {
        "expected": "Exxonmobil",
        "user": """Document:
Page 4
Unsettled Science
Knowing that weather forecasts are reliable for a
Moreover, computer models relied upon
few days at best, we should recognize the enor-
by climate scientists predict that lower atmos-
mous challenge facing scientists seeking to pre-
pheric temperatures will rise as fast as or faster
dict climate change and its impact over the next
than temperatures at the surface. However, only
century. In spite of everyone's desire for clear
within the last 20 years have reliable global
answers, it is not surprising that fundamental
measurements of temperatures in the lower at-
gaps in knowledge leave scientists unable to
mosphere been available through the use of
make reliable predictions about future changes.
satellite technology. These measurements show
A recent report from the National Re-
little if any warming.
search Council (NRC) raises important issues,
Even less is known about the potential
including these still-unanswered questions:
positive or negative impacts of climate change.
(1) Has human activity al-
In fact, many academic
ready begun to change
Sargasso Sea Temperature
studies and field experi-
temperature and the cli-
78-
ments have demonstrated
mate, and (2) How signifi-
77
Medieval
that increased levels of car-
cant will future change be?
varm period
76 -
Little
bon dioxide can
The NRC report con-
ice age
crop and forest growth.
firms that Earth's surface
So, while some argue
temperature has risen by
that the science debate is
about 1 degree Fahrenheit
73
settled and governments
over the past 150 years.
should focus only on near-
Some use this result to
72
71 .
OF
term policies-that is empty
claim that humans are
rhetoric. Inevitably, future
causing global warming,
70 -
scientific research will help
and they point to storms or
1000
500
0
- B.C. A.D. -
500 1000 1500 2000
us understand how human
floods to say that danger-
actions and natural climate
ous impacts are already
Source: Science (1996)
change may affect the world
under way. Yet scientists remain unable to con-
and will help determine what actions may be de-
firm either contention.
sirable to address the long-term.
Geological evidence indicates that climate
Science has given us enough information
and greenhouse gas levels experience significant
to know that climate changes may pose long-
natural variability for reasons having nothing to
term risks. Natural variability and human activity
do with human activity. Historical records and
may lead to climate change that could be signif-
current scientific evidence show that Europe and
North America experienced a medieval warm
cant and perhaps both positive and negative.
Consequently, people, companies and govern-
period one thousand years ago, followed cen-
ments should take responsible actions now to
turies later by a little ice age. The geological
address the issue.
record shows even larger changes throughout
One essential step is to encourage devel-
Earth's history. Against this backdrop of large,
opment of lower-emission technologies to meet
poorly understood natural variability, it is impos-
our future needs for energy. We'll next look at
sible for scientists to attribute the recent small
the promise of technology and what is being
surface temperature increase to human causes.
done today.
ExonMobil'
www.exxon.mobil.com
2000 Exxon Mobil Corporation
Source: https://www.industrydocuments.ucsf.edu/docs/ttw10228

Question: Which company name is mentioned at the bottom?""",
    },
    "docvqa_val_2": {
        "expected": "18 million",
        "user": """Document:
Page 4
DOMESTIC PRODUCT DEVELOPMENT (cont'd.)
POL 0911, B&H Menthol versus Salem 100 - B&H Menthol, without print down rod, are
being produced in Cabarrus this week.
HTI 1723, Marlboro Lights Menthol versus Salem Lights 100's samples are being
produced in Louisville this week.
Market Research
HTI 2526 and HTI 2532, Marlboro 80 Box versus Camel 80 Box - These samples have
been approved for shipment on 6/4/90.
INTERNATIONAL PRODUCT DEVELOPMENT
PM Super Lights (Hong Kong)
Production start-up of Philip Morris Super Lights Menthol began the 6th of June at the
Manufacturing Center. The 18 million order is to be shipped to Hong Kong in preparation for a
July 1 launch.
Project Ring (Korea)
Cigarettes for PMI test #13 (Parliament KS 9mg versus 88 Lights) have been approved and
shipped to the warehouse.
Seoul Consumer Panel Testing (Korea)
Cigarettes for SCP #9 (88 Lights versus PM Super Lights carbon loading study) have been
approved and shipped to the warehouse. Filters have been made and combined for SCP #10
Parliament filter study).
Merit Lights (Hong Kong)
Cigarettes for PMI testing of Merit Lights prototype versus Kent have been produced and
are under analysis.
4
2022155854
Source: https://www.industrydocuments.ucsf.edu/docs/khxj0037

Question: how much order is to be shipped to hong kong?""",
    },
    "docvqa_val_3": {
        "expected": "Philip Morris Super Lights",
        "user": """Document:
Page 4
DOMESTIC PRODUCT DEVELOPMENT (cont'd.)
POL 0911, B&H Menthol versus Salem 100 - B&H Menthol, without print down rod, are
being produced in Cabarrus this week.
HTI 1723, Marlboro Lights Menthol versus Salem Lights 100's samples are being
produced in Louisville this week.
Market Research
HTI 2526 and HTI 2532, Marlboro 80 Box versus Camel 80 Box - These samples have
been approved for shipment on 6/4/90.
INTERNATIONAL PRODUCT DEVELOPMENT
PM Super Lights (Hong Kong)
Production start-up of Philip Morris Super Lights Menthol began the 6th of June at the
Manufacturing Center. The 18 million order is to be shipped to Hong Kong in preparation for a
July 1 launch.
Project Ring (Korea)
Cigarettes for PMI test #13 (Parliament KS 9mg versus 88 Lights) have been approved and
shipped to the warehouse.
Seoul Consumer Panel Testing (Korea)
Cigarettes for SCP #9 (88 Lights versus PM Super Lights carbon loading study) have been
approved and shipped to the warehouse. Filters have been made and combined for SCP #10
Parliament filter study).
Merit Lights (Hong Kong)
Cigarettes for PMI testing of Merit Lights prototype versus Kent have been produced and
are under analysis.
4
2022155854
Source: https://www.industrydocuments.ucsf.edu/docs/khxj0037

Question: full form of PM super lights""",
    },
    "docvqa_val_4": {
        "expected": "INTER-OFFICE CORRESPONDENCE",
        "user": """Document:
Page 1
PHILIP MORRIS. U. S.A.
INTER - OFFICE CORRESPONDENCE
Richmond, Virginia
To:
.Dr. Richard Carchman
Date: May 9, 1990
From:
.Maria Shulleeta
Subject:
.Prospective Alternate Preservatives List for Phase I Screening
After examining pertinant literature and discussing with knowledgeable PM personnel the
company's continuing need for an alternate preservative for the RL process , a number of
compounds have been identified for screening in Phase I preservative assays. Some of these
compounds are known tobacco constituents whose structures are similiar to other compounds
which have demonstrated significant antimicrobial activity in our assays. Other compounds on
the proposed list are essential oils or essential oil components which are known to have
antimicrobial activity in other test systems. The prospective test compounds are listed below
with their CAS numbers (where known). Please comment on the acceptability of the use of
these compounds in our processes. It is important to consider that any compound that is would
have to be effective (complete inhibition of bacterial growth for 24 hours) at low dose (<300
ug/ml) in Phase I screening before subsequent testing in the Phase III fermentor-scale assay or
subjective screening would be suggested. In evaluating the listed compounds, please indicate a
priority for screening by rating the compounds for acceptability (e.g., very acceptable
Mono
compounds would be rated "1" and consequently tested first):
CA
18
RTECS:
Compound
CAS number
HSAB
ATECS
MDNO Caryophyllene
87-44-5
V
Sclareol
515-03-7 7
Sclareolide
564-20-5
HSOB
RTECS
Fumaric Acid
X
110-82-2 110-17-8
Taxnets
2-phenylethyl valerate
7460-74-4 /
Send to OMaria
HSDP
Moro Phenyl acetic acid
103-82-2. J
Abietic acid
514-10-3 /
# 1902
KTECS
Xanthophyll
127- 440 -2
RTECS
MciJo Basil oil
8015-73-4
RTECS MONO: Bay oil
8006- 78-8
ASDe
PTECS
MONO Cumin oil."
8014-13-9 7
RTECS
MONO Lemongrass oil
8007-02-1
ITECS
1. 1010
Caraway oil
8000-42-8 /
H.DB
RTECE
MONO
Orange oil
208- 57-9
Mero Oakmoss oil
9000-50-4
VTECS MONO Phenylacetaldehyde
122-78:1
2022156519
Source: https://www.industrydocuments.ucsf.edu/docs/ljxj0037

Question: What kind of a communication/letter  is this?""",
    },
}

