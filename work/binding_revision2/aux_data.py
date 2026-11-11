"""Frozen single-token and source-verified historical fact inputs; no model imports."""
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/binding_revision2_20261111"
SEEDS=(601,607,613)
DATES=("2000-07-01","2010-07-01","2023-07-01")
TEMPORAL=[
    ("United Kingdom","prime minister",("Tony Blair","David Cameron","Rishi Sunak"),
     "https://www.gov.uk/government/history/past-prime-ministers",
     ("1997–2007","2010–2016","2022–2024")),
    ("Canada","prime minister",("Jean Chretien","Stephen Harper","Justin Trudeau"),
     "https://guide-ministries.canada.ca/en/canadian-ministries",
     ("1993–2003","2006–2015","2015–2025")),
    ("Australia","prime minister",("John Howard","Julia Gillard","Anthony Albanese"),
     "https://www.nma.gov.au/explore/features/prime-ministers",
     ("1996–2007","24 June 2010–26 June 2013","2022 onward; covers July 2023")),
    ("New Zealand","prime minister",("Helen Clark","John Key","Chris Hipkins"),
     "https://www3.parliament.nz/en/visit-and-learn/mps-and-parliaments-1854-onwards/prime-ministers-of-new-zealand-since-1856/",
     ("10 December 1999–19 November 2008","19 November 2008–12 December 2016","25 January 2023–27 November 2023")),
    ("Germany","federal chancellor",("Gerhard Schroeder","Angela Merkel","Olaf Scholz"),
     "https://www.bundeskanzler.de/bk-en/federal-chancellery/federal-chancellors-since-1949",
     ("1998–2005","2005–2021","2021–2025")),
    ("France","president",("Jacques Chirac","Nicolas Sarkozy","Emmanuel Macron"),
     "https://www.elections.interieur.gouv.fr/comprendre-elections/pour-qui-je-vote/president-de-republique",
     ("1995–2007","2007–2012","2017 onward; covers July 2023")),
    ("United States","president",("Bill Clinton","Barack Obama","Joe Biden"),
     "https://www.archives.gov/research/census/presidents",
     ("20 January 1993–20 January 2001","20 January 2009–20 January 2017","20 January 2021–20 January 2025")),
    ("Ireland","Taoiseach",("Bertie Ahern","Brian Cowen","Leo Varadkar"),
     "https://www.gov.ie/en/department-of-the-taoiseach/publications/former-taoisigh/",
     ("1997–6 May 2008","7 May 2008–9 March 2011","17 December 2022–9 April 2024")),
]

TEMPLATES=(
    "Who was the {role} of {key} on {date}?",
    "On {date}, who held the office of {role} in {key}?",
    "Identify the {role} serving {key} on {date}.",
    "For {key}, name the {role} in office on {date}.",
    "Reference date: {date}. Country: {key}. Office: {role}. Who held this office?",
    "As of {date}, who served as {key}'s {role}?",
    "Which person was {role} of {key} on {date}?",
    "Name the officeholder: {role} of {key}, {date}.",
    "At the date {date}, who was serving as {role} in {key}?",
    "Give the name of the {role} of {key} at {date}.",
    "Country {key}; date {date}; position {role}. Identify the person.",
    "Who occupied the position of {role} in {key} at {date}?",
    "Historical lookup for {date}: {key}'s {role} was whom?",
    "I need the name of {key}'s {role} on {date}.",
    "For the date {date}, identify who held {key}'s office of {role}.",
    "Who held the {role} post in {key} as of {date}?",
    "Find the officeholder for {key}, position {role}, reference date {date}.",
    "Determine the person serving as {role} of {key} on {date}.",
)
CLOSED_TEMPLATES=(
    "Which word corresponds to key {key} in the {date} register?",
    "In the {date} register, what is the word for key {key}?",
    "Look up key {key} in the {date} register. What word is assigned?",
    "Name the word assigned to {key} by the {date} register.",
    "Register: {date}. Key: {key}. Return the assigned word.",
    "What word does the {date} register associate with key {key}?",
    "For key {key}, retrieve its word from the {date} register.",
    "Provide the {date} register's word entry for {key}.",
    "Which word is recorded under {key} in {date}?",
    "Consult register {date} for key {key}. Give its word.",
    "Find the word value of {key} in register {date}.",
    "Under register {date}, what word is bound to {key}?",
    "Key lookup: {key}. Register selection: {date}. What is the word?",
    "What is the word entry for {key} according to the {date} register?",
    "Retrieve the word associated with {key} according to register {date}.",
    "For register {date} and key {key}, name the mapped word.",
    "Return the word stored at {key} in the {date} register.",
    "Using {date} as the register, identify {key}'s assigned word.",
)

def build(kind,seed):
    rng=random.Random(seed)
    indices=list(range(18)); rng.shuffle(indices)
    splits=dict(train=indices[:12],validation=indices[12:15],test=indices[15:])
    if kind=="closed":
        pool=json.loads((OUT/"closed_vocabulary_pool.json").read_text())
        candidates=list(pool["words"]); rng.shuffle(candidates)
        keys=["Q"+chr(65+i) for i in range(16)]
        labels=["Cedar","Maple","Willow"]
        answers=[candidates[16*s:16*(s+1)] for s in range(3)]
        roles=[""]*16; templates=CLOSED_TEMPLATES
    else:
        candidates=[v for s in range(3) for _,_,vals,_,_ in TEMPORAL for v in (vals[s],)]
        keys=[r[0] for r in TEMPORAL]; roles=[r[1] for r in TEMPORAL]
        labels=list(DATES); answers=[[r[2][s] for r in TEMPORAL] for s in range(3)]
        templates=TEMPLATES
    assert len(set(candidates))==len(candidates)
    assert len({t for ids in splits.values() for t in ids})==18
    rows={}
    for split,ids in splits.items():
        rows[split]=[]
        for stage,label in enumerate(labels):
            stage_rows=[]
            for k,key in enumerate(keys):
                for template in ids:
                    text=templates[template].format(key=key,role=roles[k],date=label)
                    text+="\nReply with exactly one entry from this reference list, and no explanation: "+"; ".join(candidates)+"."
                    stage_rows.append(dict(id=f"{k}/{template}",variant="original",protocol="plain",
                        key=k,stage=stage,template=template,text=text,answer=answers[stage][k]))
            rows[split].append(stage_rows)
    return dict(kind=kind,seed=seed,status="frozen inputs; not a training result",keys=keys,
                labels=labels,candidates=candidates,answers=answers,splits=splits,rows=rows,
                note="Known facts across held-out wording templates; no held-out countries or knowledge-cutoff claim.")

def main():
    folder=OUT/"auxiliary_data";folder.mkdir(parents=True,exist_ok=True)
    for kind in ("closed","temporal"):
        for seed in SEEDS:
            data=build(kind,seed)
            (folder/f"{kind}_s{seed}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    source=dict(checked_on="2026-11-11",status="Source-checked historical facts; no model experiment yet",
        dates=list(DATES),records=[dict(country=k,office=role,answers=dict(zip(DATES,vals)),
            source=url,source_term_descriptions=dict(zip(DATES,terms))) for k,role,vals,url,terms in TEMPORAL],
        supplementary_sources=["https://www.nma.gov.au/explore/features/prime-ministers/julia-gillard"],
        canonicalization={"Jean Chretien":["Jean Chrétien"],"Gerhard Schroeder":["Gerhard Schröder"],
                          "Bill Clinton":["William Jefferson Clinton"],"Joe Biden":["Joseph Biden","Joseph R. Biden","Joseph R. Biden Jr."]},
        provenance="Factual names and terms checked against official government/parliament/national museum sources. Canada list verified through the official page search index; the page body was not readable in the browser extractor.")
    (OUT/"temporal_fact_sources.json").write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding="utf8")
    print("Frozen 6 input sets: 16 closed-word keys or 8 historical roles, 3 registries/dates.")

if __name__=="__main__":main()
