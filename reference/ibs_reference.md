# IBS Medical Reference

This file is the equivalent of OxideCoder's `api_reference.md`. It contains verified medical
facts that MUST be included in every sub-agent prompt during data generation to prevent
hallucination. All facts are sourced from peer-reviewed guidelines and open-access resources.

## Sources

- Rome IV Criteria (2016, updated)
- ACG Clinical Guideline: Management of IBS (Lacy et al., 2021)
- NICE CG61: IBS in adults (2008, updated 2017)
- BSG Guidelines on IBS (2021)
- StatPearls: Irritable Bowel Syndrome (CC-BY 4.0)
- NHS IBS pages (Open Government Licence v3.0)
- MedlinePlus IBS (Public Domain)

---

## 1. Definition and Diagnosis

### Rome IV Diagnostic Criteria for IBS
Recurrent abdominal pain, on average, at least 1 day per week in the last 3 months,
associated with 2 or more of the following:
1. Related to defecation
2. Associated with a change in frequency of stool
3. Associated with a change in form (appearance) of stool

Criteria fulfilled for the last 3 months with symptom onset at least 6 months before diagnosis.

### IBS Subtypes
- **IBS-D** (diarrhea-predominant): >25% loose/watery stools, <25% hard/lumpy stools
- **IBS-C** (constipation-predominant): >25% hard/lumpy stools, <25% loose/watery stools
- **IBS-M** (mixed): >25% loose/watery AND >25% hard/lumpy stools
- **IBS-U** (unclassified): meets IBS criteria but cannot be categorized

### What IBS is NOT
- IBS is a functional gastrointestinal disorder, NOT a structural or biochemical abnormality
- IBS does NOT cause permanent damage to the intestines
- IBS does NOT increase risk of colorectal cancer
- IBS is NOT the same as IBD (Inflammatory Bowel Disease)

## 2. Red Flag Symptoms (MUST trigger "see a doctor" response)

These symptoms suggest something other than IBS and require medical evaluation:
- Unintentional weight loss
- Rectal bleeding or blood in stool
- Onset of symptoms after age 50
- Family history of colorectal cancer, IBD, or celiac disease
- Fever
- Nocturnal symptoms that wake patient from sleep
- Progressive worsening of symptoms
- Anemia (iron deficiency)
- Palpable abdominal mass

## 3. Epidemiology

- Global prevalence: 10-15% of the population
- More common in women (2:1 ratio in Western countries)
- Most commonly diagnosed between ages 20-40
- Significant impact on quality of life and work productivity
- Many patients never seek medical care

## 4. Pathophysiology (Current Understanding)

IBS is multifactorial with no single cause:
- **Gut-brain axis dysfunction**: Altered communication between the gut and central nervous system
- **Visceral hypersensitivity**: Increased sensitivity of gut nerves to normal stimuli
- **Altered gut motility**: Abnormal contractions of intestinal muscles
- **Gut microbiome changes**: Alterations in intestinal bacteria composition
- **Post-infectious**: Can develop after gastroenteritis (post-infectious IBS)
- **Psychological factors**: Stress, anxiety, and depression can exacerbate symptoms
- **Genetic factors**: Some genetic predisposition identified
- **Food sensitivities**: Certain foods can trigger symptoms (FODMAPs, gluten, lactose)

## 5. Dietary Management

### Low-FODMAP Diet (Monash University Protocol)
FODMAPs = Fermentable Oligosaccharides, Disaccharides, Monosaccharides, And Polyols

**Three phases:**
1. **Elimination** (2-6 weeks): Remove all high-FODMAP foods
2. **Reintroduction** (6-8 weeks): Systematically reintroduce one FODMAP group at a time
3. **Personalization**: Long-term diet based on individual tolerances

**FODMAP categories:**
- **Fructose**: Excess in honey, apples, pears, watermelon, mango
- **Lactose**: Milk, soft cheeses, yogurt, ice cream
- **Fructans**: Wheat, rye, onion, garlic, artichoke
- **Galactans (GOS)**: Legumes, lentils, chickpeas
- **Polyols**: Sorbitol, mannitol, xylitol (stone fruits, mushrooms, cauliflower)

**Important notes:**
- The low-FODMAP diet should be supervised by a registered dietitian
- It is NOT meant to be followed long-term in its elimination phase
- Evidence supports effectiveness in 50-80% of IBS patients
- Should be combined with general healthy eating principles

### General Dietary Advice (NICE)
- Regular meal patterns, not skipping meals
- Adequate fluid intake (at least 8 cups/day)
- Limit caffeine to 3 cups/day
- Limit alcohol intake
- Limit fresh fruit to 3 portions/day
- Reduce intake of resistant starch (reheated foods)
- If bloating: consider increasing oats and linseeds

### Fiber
- **Soluble fiber** (psyllium, oats): Generally helpful, especially for IBS-C
- **Insoluble fiber** (bran, whole wheat): May worsen symptoms, especially bloating
- Increase fiber gradually to avoid worsening gas and bloating

## 6. Pharmacological Treatments

### Antispasmodics (first-line for abdominal pain)
- Hyoscine butylbromide (Buscopan)
- Peppermint oil capsules (enteric-coated)
- Dicyclomine (Bentyl)
- Mebeverine

### For IBS-D (diarrhea)
- Loperamide (Imodium) — for acute diarrhea episodes
- Rifaximin — non-absorbable antibiotic, FDA-approved for IBS-D
- Eluxadoline (Viberzi) — mixed opioid receptor agonist/antagonist
- Alosetron (Lotronex) — 5-HT3 antagonist, restricted use (women with severe IBS-D)
- Bile acid sequestrants (cholestyramine) — for bile acid-related diarrhea

### For IBS-C (constipation)
- Linaclotide (Linzess) — guanylate cyclase-C agonist
- Lubiprostone (Amitiza) — chloride channel activator
- Plecanatide (Trulance) — guanylate cyclase-C agonist
- PEG-based osmotic laxatives

### Neuromodulators (for pain and global symptoms)
- **Tricyclic antidepressants (TCAs)**: Amitriptyline, nortriptyline — low dose for visceral pain
- **SSRIs**: Fluoxetine, sertraline — when anxiety/depression co-exist
- **SNRIs**: Duloxetine — for pain-predominant symptoms

**CRITICAL**: The model should NEVER recommend specific dosages. Always state
"your doctor will determine the appropriate medication and dosage for you."

## 7. Psychological Therapies

Evidence-based psychological interventions:
- **Cognitive Behavioral Therapy (CBT)**: Strong evidence, can be delivered online
- **Gut-directed hypnotherapy**: Strong evidence, especially for refractory IBS
- **Mindfulness-based stress reduction (MBSR)**: Moderate evidence
- **Psychodynamic therapy**: Some evidence

## 8. Lifestyle Management

- Regular physical activity (30 min moderate exercise, most days)
- Stress management techniques
- Adequate sleep (7-9 hours)
- Yoga and relaxation exercises
- Keeping a food and symptom diary to identify triggers

## 9. Probiotics

- Some evidence for specific strains:
  - **Bifidobacterium infantis 35624** — strongest evidence, particularly for global IBS symptoms
  - **Lactobacillus plantarum DSM 9843** (also known as 299v) — studied for abdominal pain and bloating
  - **Bifidobacterium bifidum MIMBb75** — studied for global IBS symptom improvement
- Effects are strain-specific — not all probiotics help IBS
- Trial period of 4 weeks recommended; stop if no improvement
- Dose and strain matter — general "probiotic" recommendations are insufficient

## 10. Bristol Stool Chart

- **Type 1**: Separate hard lumps (severe constipation)
- **Type 2**: Lumpy, sausage-shaped (mild constipation)
- **Type 3**: Sausage with cracks on surface (normal)
- **Type 4**: Smooth, soft sausage (normal, ideal)
- **Type 5**: Soft blobs with clear edges (lacking fiber)
- **Type 6**: Mushy, fluffy pieces (mild diarrhea)
- **Type 7**: Entirely liquid (severe diarrhea)

IBS-D: predominantly Type 6-7
IBS-C: predominantly Type 1-2
