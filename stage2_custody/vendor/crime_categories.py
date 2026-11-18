"""Группировка c_charge_desc (513 формулировок) в укрупнённые типы преступлений.

Правила проверяются по порядку, срабатывает первое совпадение (регистр не важен).
Порядок важен: например, «Burglary Dwelling Assault/Batt» относится к насильственным,
«DUI Property Damage» к транспортным, «Grand Theft Firearm» к оружию.
"""
import re

import pandas as pd

RULES = [
    ("NoCharge", r"no charge"),
    ("Sex", r"lewd|lasciv|sex|prostitu|indecent expos|voyeur|porn|obscen|molest|travel.*minor|child exploit|computer for child"),
    ("Traffic", r"\bdui\b|d\.u\.i|\bdwi\b|\bdwls\b|driv|licen|\bdl\b|\blic\b|motorcycle|vehicle insurance|motor veh ins"
                r"|insur.*card|proof of invalid insur|flee|elud|leav.*(scen|acc)|lve/scen|reckless driv|register vehicle"
                r"|tag|sticker|boater|load|tri-rail|taxi"
                # нарушения ПДД (часто идут сопутствующими обвинениями в том же деле)
                r"|safety belt|stbelt|seat ?belt|child restraint|pers/inj/prot|submit insurance"
                r"|stop sign|signal|red light|flashing|yield|ftyrow|\bspeed|too fast|following too closely"
                r"|\blanes?\b|\bturn|backing|stop/stand|stop stand|stand/park|imp park|improp park|park handicap|park/obstruct|hov|sunscr|wndow|windshield|window"
                r"|due care|unsafe/improp|unsafe/improper|improp equip|lights|lamps|mcycle|bike|ped fail|x-walk"
                r"|refuse submit blood|breath test|render aid|rept accd|road surface|sounds>1000|display regist"),
    ("Violent", r"murder|manslaughter|homicide|home invasion|kidnap|carjack|robbery|battery|\bbatt\b|batt/|assault|false imprison|stalk"
                r"|child abuse|cruelty|neglect|abuse|shoot|throw.*missile|throw in occup|harm public servant"
                r"|injunc|dom vi|domestic|dating viol|threat|interference with custody|culpable negligence"
                r"|exposes culpable"),
    ("Drug", r"cocaine|cannabis|heroin|meth|oxycod|hydrocod|hydromorph|alprazolam|clonazepam|diazepam|lorazepam"
             r"|amphet|morphine|codeine|fentanyl|methadone|bupren|carisoprodol|phentermine|oxymorph|mdma|lsd"
             r"|ecstasy|ethylone|butylone|xlr11|pyrrolid|piperaz|pipe\b|controlled? subst|contr sub|control subst"
             r"|cont subs|paraphernalia|tetrahydro|\brx\b|prescript|drug|traff|jwh|fluoro|gamma-hydroxy|butanediol"
             r"|steroid|trenbolon|amobarbital|methox|contrft schd|countrft|cntrft|canbs|fubinaca|synthetic"),
    ("Weapon", r"firearm|weapon|\bwep\b|f/arm|concealed|ammun|destruct device|bomb"),
    ("Fraud", r"forg|utter|fraud|credit card|\bcc\b|personal id|identity|false name|false own|counterfeit|worthless"
              r"|defraud|launder|structuring|scanning device|innkeeper|ID card|unlaw.*\bid\b|unlaw use false"
              r"|imperson|police badge|simulation|draft|trade secret|intellectual|bribery|telemarket"
              r"|white collar|bad check|false info|false 911|false crime|giving false|posses.*altered bills|unlaw pos of prson|hire|lease"),
    ("Property", r"theft|burgl|stolen|mischief|shoplift|arson|trespass|tresspass|retail|cart|tools|fire extin|\bburg\b|burg/"
                 r"|altered id|alt vin|railroad|fuel|deprive"),
]
COMPILED = [(name, re.compile(rx, re.I)) for name, rx in RULES]
OTHER = "Other"
UNKNOWN = "Unknown"


def categorize(desc):
    if pd.isna(desc) or not str(desc).strip():
        return UNKNOWN
    for name, rx in COMPILED:
        if rx.search(str(desc)):
            return name
    return OTHER


def mapping_table(series):
    vc = series.value_counts()
    return pd.DataFrame({"c_charge_desc": vc.index, "rows": vc.values,
                         "category": [categorize(d) for d in vc.index]})


# ------------------------------------------------------------------------------
# Дробление Other (решение Greer, 08.10.2026). Применяется только к обвинениям,
# которые правила выше отнесли к Other. Сначала исправления явных ошибок
# группировки (обвинение относится к уже существующему типу), затем новые
# состояния. Порядок важен: срабатывает первое совпадение.
FIX_RULES = [
    ("Traffic", r"interlock device|writ acc rpt"),
    ("PublicOrder", r"harassing phone|phone ring repeat"),
    ("Property", r"tresps"),
    ("Justice", r"escape|contraband|intro cont|extradition|contempt|fail.*appear|handcuff key|elec monitor"
                r"|dna sample|fail register|failure to register|change address/name|report change/residence"
                r"|accessory after|perjury|false affidavit|false report|false reports|false id given to leo"
                r"|post bond|viol court order"),
    ("Tampering", r"tamper|witness|victim|harass|fabricat|retaliat"),
    ("Minors", r"minor|delinq|juvenile|compulsory|school function|child<6|crlty twrd child"),
    ("Conspiracy", r"conspir|crim cnsp|solicitation on|attempt/solicitation|criminal attempt|principal in the first|2 way device"),
    ("Drug", r"sale/del/pos|temazepam|pethidine|benocy|psilocyn|s/m/d/p|cnsp traf|halluc|cann|contrl sub|contr sub"
             r"|marijuana|chlordiazepox|pb-22|nbome|ur-144|propoxyphene|paraph|flurazepam|testost|w/int near a park"
             r"|inhale harmful"),
    ("Fraud", r"corrupt public servant|pawn|secondhand|reg metals|exploit elderly|cred|false receipt|counterf|false insurance"
              r"|false statemnt ins|false personation|false documents|rico|patient brokering|money trans|currenc|bank bills"
              r"|copy of sound|audio visual|sound articles|vin plate|altered motor vehicle|id\" removed|alter id markings"
              r"|work comp ins|computer user|personating officer"),
    ("Property", r"coin operated|copper|phone cables|phone/power|petit frm merchant|property not structure|fence"
                 r"|dairy case|altering \(outdoor advert\)"),
    ("Weapon", r"f/a serial|f/a altered|short barl|self propelled knife"),
    ("Violent", r"affray|engage in fighting|fight/brawl|agg bodily inj|att murd|baiting animals|hiv|dog bite"),
    ("Traffic", r"imp start|engine run|proof of insur|airless tires|improper pass|improper stop|improp|unauthorizd person drv"
                r"|headset|improper parking|ped|walk left|move over|block intersection|disobey green|double parked|muffler"
                r"|stop lamp|moped|special haz|fast for cond|horn|atv|lghts|rear view|passing|no pass|bicycle|school bus"
                r"|no d/l|rear light|rear lamp|windscreen|exhaust|defective equip|bumper|veh reg|safety +belt|accident report"
                r"|accid|habit traf|head lamp|headlamp|^infraction$|unsafe equip|too loud from vehicle|cmv|rr barrier|rr crossing"
                r"|lim acces|obstrct veh|crash|motor vehicle noise|operate motor vehicle|veh on public beach|100' of intersection"
                r"|financial responsibility|racing|upper beam|vehicle dealer|aircraft registration"),
    ("ResistPolice", r"resist|obstruct|obey police|disobey officer|refuse stop when ordered|depriv leo|k9|police"),
    ("PublicOrder", r"prowl|loiter|disorderly|doc/|bev|alcohol|alch|container|cont in m/v|cntnr|contnr|drink|drunk|intox"
                    r"|911|false alarm|sleep|lodg|litter|liter|urinate|panhandl|solic|noise|gambl|bookmak|lottery|ordinance"
                    r"|unlawful assembly|disturbing the peace|disturb education|city park|park hours|beach|indecen|wear mask"
                    r"|vaccine|plumbing|construct contract|mobile unit|guests|bigamy|optical dev|desertion|electrical|county"),
]
FIX_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in FIX_RULES]
NEW_STATES = ["ResistPolice", "PublicOrder", "Tampering", "Justice", "Conspiracy", "Minors"]


def categorize_fine(desc):
    base = categorize(desc)
    if base != OTHER:
        return base
    for name, rx in FIX_COMPILED:
        if rx.search(str(desc)):
            return name
    return OTHER
