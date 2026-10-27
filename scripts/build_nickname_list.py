"""Builds the default star-shaped nickname list (nicknames.csv) from carltonnorthern/nicknames.

Usage: python scripts/build_nickname_list.py <path to the source names.csv> [output path]

The source lists "name1,has_nickname,name2" rows in both directions and with many standard
names per nickname. The build keeps one standard name per nickname:
  - a name that is a nickname of a name with more nicknames of its own is itself a nickname,
    so reverse rows (bob -> robert) and chains (bobby -> bob) collapse onto the fuller name;
    the standard names the curated tables point at are never demoted;
  - a name that stays standard is never kept as a nickname of another name;
  - where several standard names remain for a nickname, a curated preference for the most common
    one wins, otherwise the standard name with the most nicknames, then the alphabetically first;
  - nicknames the source lacks are added from ADDED_NICKNAMES.
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PREFERRED_STANDARD_NAME = {
    "abbie": "abigail", "abby": "abigail", "al": "albert", "alex": "alexander", "andy": "andrew", "ben": "benjamin",
    "benny": "benjamin", "bernie": "bernard", "betsy": "elizabeth", "betty": "elizabeth",
    "bill": "william", "billy": "william", "bob": "robert", "bobby": "robert", "chuck": "charles",
    "dan": "daniel", "danny": "daniel", "dave": "david", "davey": "david", "dick": "richard",
    "ed": "edward", "eddie": "edward", "ned": "edward", "ted": "edward", "teddy": "edward",
    "fred": "frederick", "freddie": "frederick", "freddy": "frederick", "geoff": "geoffrey",
    "chris": "christopher", "jon": "jonathan", "jonny": "jonathan", "jack": "john", "jake": "jacob", "jay": "jason", "len": "leonard", "lenny": "leonard", "jeff": "jeffrey", "jen": "jennifer", "jenny": "jennifer",
    "jim": "james", "jimmy": "james", "joe": "joseph", "johnny": "john", "kate": "katherine",
    "katie": "katherine", "kathy": "katherine", "ken": "kenneth", "larry": "lawrence",
    "liz": "elizabeth", "maggie": "margaret", "matt": "matthew", "matty": "matthew",
    "mattie": "matthew", "meg": "margaret", "mickey": "michael", "mike": "michael",
    "mikey": "michael", "nate": "nathan", "nick": "nicholas", "pat": "patrick", "peggy": "margaret",
    "rick": "richard", "rob": "robert", "robbie": "robert", "ron": "ronald", "ronnie": "ronald",
    "sam": "samuel", "sammy": "samuel", "steve": "steven", "tim": "timothy", "tom": "thomas",
    "tommy": "thomas", "tony": "anthony", "vince": "vincent", "will": "william", "willy": "william",
    "zach": "zachary", "zack": "zachary",
}

ADDED_NICKNAMES = {
    "matty": "matthew", "mattie": "matthew", "davie": "david", "danny": "daniel", "dani": "daniel",
    "jimmie": "james", "jamie": "james", "joey": "joseph", "joe": "joseph", "tommie": "thomas",
    "willie": "william", "liam": "william", "robbie": "robert", "robby": "robert", "ricky": "richard",
    "richie": "richard", "rich": "richard", "dick": "richard", "rick": "richard", "mikey": "michael",
    "mick": "michael", "mikee": "michael", "andie": "andrew", "drew": "andrew", "tonie": "anthony",
    "tony": "anthony", "steevie": "steven", "stevie": "steven", "steve": "steven", "chrissie": "christine",
    "chris": "christopher", "topher": "christopher", "kit": "christopher", "alex": "alexander",
    "xander": "alexander", "sandy": "alexander", "lex": "alexander", "ally": "alison", "kenny": "kenneth",
    "ken": "kenneth", "greg": "gregory", "gregg": "gregory", "jon": "jonathan", "jonny": "jonathan",
    "benji": "benjamin", "ben": "benjamin", "sam": "samuel",
    "sammy": "samuel", "charlie": "charles", "chuck": "charles", "chas": "charles", "hank": "henry",
    "harry": "henry", "hal": "henry", "lenny": "leonard", "len": "leonard", "phil": "philip",
    "pete": "peter", "ron": "ronald", "don": "donald", "donnie": "donald", "ted": "edward",
    "teddy": "edward", "eddie": "edward", "ed": "edward", "ned": "edward", "kate": "katherine",
    "katie": "katherine", "kathy": "katherine", "kat": "katherine", "kitty": "katherine",
    "liz": "elizabeth", "lizzie": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "libby": "elizabeth", "jen": "jennifer", "jenny": "jennifer", "jess": "jessica", "jessie": "jessica",
    "becky": "rebecca", "becca": "rebecca", "sue": "susan", "susie": "susan", "suzy": "susan",
    "meg": "margaret", "maggie": "margaret", "peggy": "margaret", "maddie": "madeline",
    "abby": "abigail", "tina": "christina", "cindy": "cynthia", "debbie": "deborah", "deb": "deborah",
    "patty": "patricia", "trish": "patricia", "pam": "pamela", "sally": "sarah", "molly": "mary",
    "polly": "mary", "annie": "anne", "nan": "anne", "vicky": "victoria", "tori": "victoria",
}


def read_source_rows(source_path: Path) -> list[tuple[str, str]]:
    """Return the (standard name, nickname) pairs of the source file, lower-cased."""
    with source_path.open(newline="", encoding="utf-8") as handle:
        return [
            (row["name1"].strip().lower(), row["name2"].strip().lower())
            for row in csv.DictReader(handle)
            if row["relationship"] == "has_nickname"
        ]


def build_star_list(source_rows: list[tuple[str, str]]) -> dict[str, str]:
    """Collapse source pairs into {nickname: one standard name}."""
    nicknames_of: dict[str, set[str]] = defaultdict(set)
    for standard, nickname in source_rows:
        if standard != nickname:
            nicknames_of[standard].add(nickname)
    own_count = Counter({name: len(found) for name, found in nicknames_of.items()})

    parents_of: dict[str, set[str]] = defaultdict(set)
    for standard, found in nicknames_of.items():
        for nickname in found:
            parents_of[nickname].add(standard)

    protected = set(PREFERRED_STANDARD_NAME.values()) | set(ADDED_NICKNAMES.values())
    demoted = {
        name
        for name in nicknames_of
        if name not in protected and any(own_count[parent] > own_count[name] for parent in parents_of[name])
    }
    standards = set(nicknames_of) - demoted

    mapping: dict[str, str] = {}
    for nickname in set(parents_of) | demoted:
        if nickname in standards:
            continue
        candidates = sorted(parents_of[nickname] & standards, key=lambda name: (-own_count[name], name))
        if candidates:
            mapping[nickname] = candidates[0]

    for nickname, standard in ADDED_NICKNAMES.items():
        if nickname not in mapping and nickname not in standards and nickname not in protected:
            mapping[nickname] = standard
    for nickname, standard in PREFERRED_STANDARD_NAME.items():
        if nickname not in protected:
            mapping[nickname] = standard
    for nickname in [name for name, standard in mapping.items() if standard in mapping]:
        mapping[nickname] = mapping[mapping[nickname]]
    return mapping


def main() -> None:
    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "src" / "pii_toolbox" / "coref" / "nicknames.csv"
    mapping = build_star_list(read_source_rows(source_path))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for nickname in sorted(mapping):
            writer.writerow([nickname, mapping[nickname]])


if __name__ == "__main__":
    main()
