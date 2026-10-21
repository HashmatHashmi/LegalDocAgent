"""Rule library, missing-clause checks and semantic prototypes for risk detection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple

F = re.IGNORECASE


@dataclass
class Rule:
    id: str
    category: str
    severity: str  # High | Medium | Low
    pattern: "re.Pattern"
    explanation: str
    suggestion: str
    escalate: Optional["re.Pattern"] = None
    applies: Set[str] = None  # type: ignore[assignment]
    superseded_by: Tuple[str, ...] = ()


def R(id, category, severity, pattern, explanation, suggestion, escalate=None, applies=("any",), superseded_by=()):
    return Rule(id, category, severity, re.compile(pattern, F), explanation, suggestion,
                re.compile(escalate, F) if escalate else None, set(applies), tuple(superseded_by))


RULES = [
    R("indemnity", "Indemnification / hold harmless", "Medium",
      r"\b(?:indemnif(?:y|ies|ied|ication)|(?:hold|held|holds)\s+(?:\w+\s+){0,3}harmless)\b",
      "An indemnity shifts the cost of third-party claims and losses onto one party; if one-sided or uncapped it creates open-ended exposure.",
      "Make the indemnity mutual, limit it to losses caused by the indemnifying party's own breach or negligence, exclude the other side's fault and add a liability cap.",
      escalate=r"\b(?:all|any|every)\s+(?:\w+\s+){0,3}(?:claims?|losses|damages|liabilit\w*|costs|expenses)|whatsoever|without\s+limit|regardless\s+of|own\s+negligence"),
    R("unlimited_liability", "Unlimited / uncapped liability", "High",
      r"\bunlimited\s+liabilit\w*|\bwithout\s+(?:any\s+)?(?:limit|limitation|cap)\b|\bliable\s+for\s+(?:all|any|every)\b|\bfull\s+liability\b",
      "Liability is not capped, so a single claim could far exceed what the contract is worth.",
      "Cap liability at a defined amount (for example fees paid in the previous 12 months) and exclude indirect losses."),
    R("one_sided_termination", "One-sided termination right", "High",
      r"\b(?:may|can|shall\s+have\s+the\s+right\s+to|reserves?\s+the\s+right\s+to|is\s+entitled\s+to)\s+(?:immediately\s+)?terminat\w*[^.;]{0,100}?"
      r"(?:at\s+any\s+time|without\s+(?:any\s+)?(?:cause|reason|notice|liability)|for\s+any\s+reason|sole\s+discretion|at\s+its\s+(?:own\s+)?discretion|for\s+convenience)",
      "One party can end the agreement freely, leaving the other without security or time to react.",
      "Give both parties the same termination rights, require reasonable written notice (e.g. 30 days) and allow a cure period for breaches."),
    R("no_notice", "Action without notice", "Medium",
      r"\bwithout\s+(?:any\s+|prior\s+|written\s+|advance\s+|further\s+)*(?:notice|warning|intimation)\b",
      "A right can be exercised without warning, so the affected party cannot prepare, object or remedy the issue.",
      "Require prior written notice (e.g. 30 days, or at least 7 days for breaches) before the right can be exercised."),
    R("sole_discretion", "Unilateral / sole discretion", "Medium",
      r"\b(?:sole|absolute|unfettered|exclusive)\s+(?:and\s+\w+\s+)?discretion\b|\bas\s+(?:it|he|she|they|the\s+\w+)\s+(?:deems?|sees?|thinks?)\s+(?:fit|appropriate|necessary)\b",
      "A decision is left entirely to one party with no objective standard or review.",
      "Replace with 'reasonable discretion exercised in good faith' or tie the decision to objective criteria."),
    R("unilateral_change", "Unilateral modification of terms", "High",
      r"\b(?:may|can|reserves?\s+the\s+right\s+to|entitled\s+to)\s+(?:\w+\s+){0,2}(?:amend|modify|change|revise|vary|alter|update)\b[^.;]{0,80}?"
      r"(?:at\s+any\s+time|from\s+time\s+to\s+time|without\s+(?:prior\s+)?(?:notice|consent)|sole\s+discretion|unilateral\w*)",
      "One party can rewrite the contract after signing, which undermines the agreed deal.",
      "Require changes to be in writing and signed by both parties (or give advance notice plus a right to exit)."),
    R("price_increase", "Unilateral price / rent increase", "High",
      r"\b(?:increase|raise|revise|escalate|enhance)\w*\s+(?:\w+\s+){0,3}(?:rent|fees?|price|charges?|rates?|remuneration)\b[^.;]{0,80}?"
      r"(?:at\s+any\s+time|sole\s+discretion|without\s+(?:prior\s+)?(?:notice|consent)|from\s+time\s+to\s+time|unilateral\w*)",
      "Costs can be raised at will, making the total price unpredictable.",
      "Fix the amount or a formula (e.g. capped annual increase of a stated %) and require advance written notice."),
    R("auto_renewal", "Automatic renewal / lock-in", "Medium",
      r"\bautomatic(?:ally)?\s+(?:renew|extend)\w*|\bauto-?renew\w*|\b(?:renew|extend)\w*\s+automatically|\bevergreen\b|\b(?:minimum|lock-?in)\s+(?:term|period)\b|\bearly\s+termination\s+(?:fee|charge|penalt\w*)",
      "The contract continues or locks you in unless you act in time, often with exit charges.",
      "Require express written renewal or a clear opt-out reminder, with a short cancellation notice and no exit penalty."),
    R("perpetual", "Perpetual / irrevocable obligation", "Medium",
      r"\b(?:perpetual(?:ly)?|in\s+perpetuity|irrevocabl[ey]|for\s+ever|forever|indefinite(?:ly)?)\b",
      "An obligation or right never ends, which can outlive the relationship.",
      "Give the obligation a fixed term (e.g. 2-5 years) and a way to terminate it.",
      superseded_by=("confidentiality_survival",)),
    R("penalty", "Penalty / liquidated damages / forfeiture", "Medium",
      r"\b(?:penalt(?:y|ies)|liquidated\s+damages|forfeit\w*)\b",
      "A fixed penalty or forfeiture may be out of proportion to the actual loss and can be unenforceable.",
      "Replace with a genuine pre-estimate of the other party's actual loss, cap the amount, and credit whatever that party recovers by mitigating its loss.",
      escalate=r"\b(?:\d{2,3}\s*%|double|triple|twice|treble|entire|full|all)\b"),
    R("late_fees", "Late fees / high interest", "Low",
      r"(?:PKR|Rs\.?|USD|\$|EUR|£)\s?[\d,]+\s*(?:per|each|every|a)\s+day|\bdaily\s+(?:fine|penalt\w*|charge|fee)\b|\blate\s+(?:fee|charge)s?|\blate\s+payment\s+(?:fee|charge|interest)|\binterest\s+(?:at|of)\s+[^.;]{0,25}(?:per\s+(?:month|annum|day)|%\s*(?:per|p\.?\s*[am]))|\b\d+(?:\.\d+)?\s*%\s*(?:per\s+(?:month|day|annum)|monthly|daily)",
      "Late-payment charges can compound quickly.",
      "Cap the interest at a reasonable rate and apply it only after a grace period."),
    R("approval_any_reason", "Consent can be refused for any reason", "Medium",
      r"\b(?:consent|approval|permission)\b[^.;]{0,60}\b(?:may|can|shall)\s+be\s+(?:withheld|refused|denied)\b[^.;]{0,30}\b(?:any|no)\s+reason\b|\bwithheld\s+for\s+any\s+reason\b|\bat\s+(?:its|his|her|their)\s+(?:own\s+)?(?:option|will|pleasure)\b",
      "One party can refuse approval arbitrarily, so you cannot rely on getting it.",
      "State that approval may not be unreasonably withheld or delayed, and set a response deadline."),
    R("take_it_or_leave_it", "'Final / non-negotiable' terms", "Low",
      r"\bnon-?\s?negotiable\b|\bnot\s+(?:be\s+)?(?:subject\s+to\s+)?negotiat\w*|\bfinal\s+and\s+binding\b",
      "Terms are declared non-negotiable, which can discourage review and may weaken your position.",
      "Remove the wording; you are entitled to negotiate and to have changes recorded in writing."),
    R("blanket_responsibility", "Responsibility regardless of cause / fault", "Medium",
      r"\bregardless\s+of\s+(?:the\s+)?(?:cause|fault|reason|who)\b|\beven\s+if\s+(?:the\s+)?(?:bills?|damage|loss|defect|failure)s?\b[^.;]{0,80}\b(?:arise|caused|result)\w*",
      "You bear the cost even when the other party caused the problem or it pre-dates you.",
      "Limit responsibility to what the party itself causes or incurs during its own period of use; the other side covers what it caused or what pre-dates the term."),
    R("subjective_trigger", "Termination/default on one party's belief", "High",
      r"\bif\s+(?:the\s+)?[A-Za-z]+\s+(?:believes|considers|deems|feels|thinks|suspects)\b[^.;]{0,80}",
      "A right is triggered by one party's opinion rather than an objective, provable event.",
      "Tie the trigger to a defined, objectively verifiable breach with written notice and a cure period."),
    R("non_compete", "Non-compete / exclusivity restriction", "Medium",
      r"\bnon-?\s?compet\w*|\bnot\s+(?:to\s+)?(?:directly\s+or\s+indirectly\s+)?(?:compete|engage\s+in\s+(?:any\s+)?(?:competing|similar))|\bexclusiv(?:e|ely|ity)\b(?!\s+(?:jurisdiction|venue|forum|remed))|\bshall\s+not\s+(?:work|be\s+employed)\s+(?:for|with)\b",
      "Restrictions on working with others can limit your livelihood or business and may be unenforceable if too broad.",
      "Limit the restriction in duration (6-12 months), geography and scope, and make it conditional on fair compensation."),
    R("non_solicit", "Non-solicitation restriction", "Low",
      r"\bnon-?\s?solicit\w*|\bshall\s+not\s+(?:directly\s+or\s+indirectly\s+)?solicit\b",
      "Prevents approaching customers or staff after the relationship ends.",
      "Limit to a short period and to clients/staff the party actually dealt with."),
    R("waiver", "Waiver of rights / claims", "Medium",
      r"\b(?:waive[sd]?|waiving|waiver)\b[^.;]{0,60}\b(?:right|rights|claims?|remed(?:y|ies)|defen[cs]es?|jury|notice|objection)\b|\brelease[sd]?\b[^.;]{0,40}\b(?:all|any)\s+claims?\b",
      "Giving up rights in advance can remove important protections.",
      "Delete the waiver or limit it to specific, clearly described claims and make it mutual."),
    R("dispute_forum", "Restrictive dispute-resolution terms", "Low",
      r"\b(?:exclusive\s+jurisdiction|(?:binding|mandatory|compulsory)\s+arbitration|class\s+action|jury\s+trial|sole\s+arbitrator\s+(?:appointed|nominated)\s+by)\b",
      "Disputes may have to be resolved in a forum that is inconvenient or controlled by the other side.",
      "Use a neutral forum or arbitrator agreed by both parties in a location convenient to both."),
    R("ip_assignment", "Broad IP assignment", "Medium",
      r"\b(?:assign|transfer|convey|vest)s?\b[^.;]{0,80}\b(?:all|any|entire)\b[^.;]{0,60}\b(?:intellectual\s+property|inventions?|work\s+product|copyrights?|rights\s+(?:and|,)\s+interests?)\b|\bwork(?:s)?\s+(?:made\s+)?for\s+hire\b"
      r"|(?:intellectual\s+property|work\s+product|inventions?|copyrights?)\b[^.;]{0,120}\b(?:(?:is|are|shall\s+be|be)\s+(?:hereby\s+)?(?:assigned|transferred|conveyed)|shall\s+(?:belong|vest)|(?:is|are)\s+(?:the\s+)?(?:sole|exclusive)\s+property|owned\s+(?:solely|exclusively))",
      "Ownership of everything created may pass to the other party, even work unrelated to the agreement.",
      "Limit the assignment to deliverables created specifically for this agreement and keep pre-existing IP with its owner."),
    R("warranty_disclaimer", "Warranty disclaimer / 'as is'", "Medium",
      r"\bas[\s-]is\b|\bdisclaim\w*\b[^.;]{0,50}\bwarrant\w*|\bwithout\s+(?:any\s+)?warrant\w*",
      "You get no guarantee of quality or fitness, so defects are your problem.",
      "Add basic warranties (quality, fitness for purpose, non-infringement) and a remedy period."),
    R("consequential", "Exclusion of consequential damages", "Low",
      r"\b(?:consequential|indirect|incidental|special|punitive)\s+(?:or\s+\w+\s+)?damages?\b",
      "Excludes recovery of knock-on losses; check that it applies equally to both parties.",
      "Make the exclusion mutual and carve out fraud, wilful misconduct and confidentiality breaches."),
    R("confidentiality_survival", "Open-ended confidentiality", "Medium",
      r"\bconfidential\w*[^.;]{0,120}\b(?:perpetual\w*|indefinite\w*|forever|in\s+perpetuity|without\s+(?:any\s+)?time\s+limit)\b|\bsurviv\w*[^.;]{0,60}\b(?:indefinite\w*|perpetual\w*|forever)\b",
      "Confidentiality duties never expire, which is hard to comply with over time.",
      "Limit confidentiality to a fixed period (e.g. 3-5 years after termination), keeping trade secrets protected for as long as they remain secret."),
    R("nonrefundable", "Non-refundable payment / deposit forfeiture", "Medium",
      r"\bnon-?\s?refundable\b|\bnot\s+(?:be\s+)?refundable\b|\b(?:deposit|advance|security)\b[^.;]{0,60}\b(?:forfeit\w*|retain(?:ed)?\s+(?:in\s+full|entirely))",
      "Money paid up-front cannot be recovered even if the other party is at fault.",
      "Make payments refundable (or pro-rata) except for documented losses, with a deadline for return."),
    R("entry_access", "Access / entry without notice", "High",
      r"\b(?:enter|access|inspect|re-?enter)\w*\b[^.;]{0,80}\b(?:at\s+any\s+time|without\s+(?:prior\s+|any\s+)?notice|whenever\s+(?:he|she|it|they)\s+(?:wishes|desires|deems))",
      "The other party can enter or inspect at will, affecting privacy and quiet enjoyment.",
      "Require at least 24-48 hours' notice, reasonable hours and a valid reason (except genuine emergencies).",
      applies=("lease",)),
    R("eviction", "Summary eviction / termination of tenancy", "High",
      r"\b(?:evict\w*|vacate\w*|re-?enter\w*|re-?possess\w*|terminate\s+(?:the\s+)?(?:lease|tenancy))\b[^.;]{0,80}\b(?:without\s+(?:any\s+)?(?:notice|court|legal|process)|immediately|forthwith|at\s+any\s+time)\b",
      "The tenant can be removed immediately or without due process.",
      "Require written notice, a cure period and lawful eviction procedure.",
      applies=("lease",)),
    R("rent_burden", "Tenant bears all repairs / damage / costs", "Medium",
      r"\b(?:tenant|lessee)\b[^.;]{0,60}\b(?:responsible|liable|bear|pay)\b[^.;]{0,50}\b(?:all|any|every)\b[^.;]{0,30}\b(?:repair|maintenance|damage|loss|tax|cost)\w*",
      "Structural repairs and ordinary wear are normally the landlord's responsibility.",
      "Limit tenant duties to minor repairs and damage it causes; landlord keeps structure, wiring and plumbing.",
      applies=("lease",)),
    R("short_deadline", "Very short cure / compliance deadline", "Medium",
      r"\bwithin\s+(?:twenty[\s-]four|forty[\s-]eight|seventy[\s-]two|24|48|72|one|two|three|1|2|3)\s*(?:\(\d+\)\s*)?(?:hours?|days?)\b(?=[^.;]{0,100}\b(?:cure|remed\w*|pay\w*|vacat\w*|terminat\w*|default|breach|forfeit\w*|comply|surrender\w*|clear\w*))",
      "The time allowed to fix a problem or pay is unrealistically short.",
      "Allow a reasonable cure period (e.g. 10-30 days) before any consequence applies."),
    R("assignment", "Assignment / subcontracting without consent", "Medium",
      r"\b(?:assign|transfer|delegate|subcontract)\w*\b[^.;]{0,80}\bwithout\b[^.;]{0,30}\b(?:consent|approval|permission|notice)\b|\bfree(?:ly)?\s+assign\w*",
      "The contract can be passed to a stranger without your approval.",
      "Require prior written consent (not unreasonably withheld) for assignment by either party."),
    R("setoff", "Set-off / withholding of payment", "Medium",
      r"\b(?:withhold|set[\s-]?off|deduct)\w*\b[^.;]{0,60}\b(?:payment|amount|sum|fee|salary|wages|remuneration|deposit)s?\b",
      "Payments can be reduced or held back at the other party's discretion.",
      "Allow deductions only for undisputed, documented amounts after written notice."),
    R("joint_several", "Joint and several liability", "Medium",
      r"\bjointly\s+and\s+severally\b|\bjoint\s+and\s+several\b",
      "Each person can be made to pay the whole debt, not just their share.",
      "Limit each party's liability to its own share or add a right of contribution."),
    R("personal_guarantee", "Personal guarantee / personal liability", "High",
      r"\bpersonal(?:ly)?\s+(?:guarant\w*|liable|responsible)\b",
      "Your personal assets are exposed if the company or other party defaults.",
      "Remove the personal guarantee or cap it at a fixed amount and period."),
    R("acceleration", "Acceleration of debt", "Medium",
      r"\baccelerat\w*\b[^.;]{0,60}\b(?:due|payable|maturity|repayment|payment)\b|\b(?:entire|whole|full|outstanding)\s+(?:balance|amount|principal)\b[^.;]{0,50}\b(?:immediately|forthwith)\s+(?:due|payable)\b|\bimmediately\s+due\s+and\s+payable\b",
      "A single default can make the whole amount payable at once.",
      "Allow acceleration only after written notice and a cure period for material defaults."),
    R("confession_judgment", "Blank / pre-signed instruments", "High",
      r"\bconfession\s+of\s+judg\w*|\bpre-?signed\b|\bblank\s+(?:cheque|check|signed|stamp)\w*|\bundated\s+cheque",
      "Blank or pre-signed instruments can be misused to claim more than is owed.",
      "Delete the clause; use specified, dated payment instruments only."),
    R("service_bond", "Service bond / training cost recovery", "Medium",
      r"\bservice\s+bond\b|\bbond\s+(?:period|amount)\b|\b(?:recover|reimburse|refund)\w*[^.;]{0,40}\b(?:training|education|course)\s+(?:cost|expense|fee)s?\b|\bliquidated\s+damages\b[^.;]{0,50}\b(?:resign|leav|quit)\w*",
      "Leaving early triggers payments that can trap the employee in the job.",
      "Limit recovery to documented training costs on a declining pro-rata basis.",
      applies=("employment",)),
    R("unpaid_work", "Uncompensated or open-ended duties", "Low",
      r"\b(?:without|no)\s+(?:any\s+)?(?:additional|extra|overtime)\s+(?:pay|payment|compensation|remuneration)\b|\bany\s+other\s+duties\b|\bas\s+(?:and\s+when\s+)?required\s+by\s+the\s+(?:company|employer)\b",
      "Duties or hours are open-ended without matching pay.",
      "Define working hours and duties and state overtime compensation.",
      applies=("employment",)),
    R("data_sharing", "Broad data collection / sharing", "Medium",
      r"\b(?:sell|share|disclose|transfer|license)\b[^.;]{0,50}\b(?:personal|user|customer)\s+(?:data|information)\b[^.;]{0,60}\b(?:third\s+part\w*|affiliates|partners|advertis\w*)\b|\bwithout\s+(?:your\s+)?(?:further\s+)?consent\b",
      "Personal data can be passed on to others without your agreement.",
      "Share data only for stated purposes, with opt-in consent and a right to delete.",
      applies=("tos", "nda")),
    R("time_essence", "Time is of the essence", "Low",
      r"\btime\s+is\s+(?:of\s+the\s+)?essence\b",
      "Any delay, however small, can count as a serious breach.",
      "Apply it only to specific critical dates and keep a short grace period."),
    R("vague_terms", "Vague / discretionary standards", "Low",
      r"\b(?:reasonable\s+efforts|best\s+efforts|as\s+may\s+be\s+determined\s+by|such\s+other\s+(?:terms|conditions)\s+as)\b",
      "Undefined standards invite disputes about what is required.",
      "Define the standard objectively (measurable targets, deadlines, deliverables)."),
    R("attorney_fees", "One-way attorney fees / costs", "Low",
      r"\b(?:attorney'?s?|legal|lawyer'?s?|advocate'?s?)\s+(?:fees?|costs?|expenses)\b[^.;]{0,80}\b(?:borne|paid|reimburs\w*|indemnif\w*)\b",
      "One party may have to pay the other's legal costs regardless of who wins.",
      "Make cost-shifting mutual and award costs to the prevailing party only."),
]

# --------------------------------------------------------------------------------------
# Missing-clause checks: (name, pattern that should be present, severity, applies, explanation, suggestion)
# --------------------------------------------------------------------------------------
MISSING_CHECKS = [
    ("Limitation of liability", r"limitation\s+of\s+liabilit|liabilit\w*[^.;]{0,80}\b(?:shall\s+not\s+exceed|limited\s+to|capped|aggregate)\b|in\s+no\s+event\s+shall",
     "Medium", {"commercial"}, "No cap on liability was found, so exposure may be unlimited.",
     "Add a mutual liability cap (e.g. fees paid in the last 12 months) with sensible carve-outs."),
    ("Governing law", r"governed\s+by|governing\s+law|laws\s+of\s+(?:the\s+)?[A-Z]|construed\s+in\s+accordance",
     "Medium", {"contract"}, "No governing-law clause was found, so it is unclear which country's law applies.",
     "State the governing law (and ideally the courts) that apply."),
    ("Dispute resolution", r"\b(?:arbitrat\w*|mediation|jurisdiction|courts?\s+of|dispute\s+resolution|venue)\b",
     "Low", {"contract"}, "No dispute-resolution process was found.",
     "Add a step-by-step process (negotiation, mediation, then arbitration or courts) and a venue."),
    ("Termination / exit", r"\bterminat\w*|\bexpir\w*|\bcancel\w*",
     "Medium", {"contract"}, "No termination or exit mechanism was found.",
     "Add termination rights for both sides, notice periods and consequences of termination."),
    ("Term / duration", r"\bterm\s+of\b|\bperiod\s+of\b|\beffective\s+(?:date|from|until)\b|\bcommenc\w*|\bfor\s+a\s+(?:term|period)\b|\bvalid\s+(?:for|until)\b|\b(?:term|lease|tenancy|agreement)\b[^.;]{0,40}\b(?:begin|start|end)s?\b|\b(?:begin|start)s?\s+on\b",
     "Low", {"contract"}, "No clear start date or duration was found.",
     "State the effective date and the duration (or how it ends)."),
    ("Confidentiality", r"confidential",
     "Low", {"employment", "commercial"}, "No confidentiality clause was found.",
     "Add a mutual confidentiality clause with a fixed term."),
    ("Force majeure", r"force\s+majeure|act\s+of\s+god|beyond\s+(?:its|their|the)\s+(?:reasonable\s+)?control",
     "Low", {"commercial", "lease"}, "No force-majeure clause was found; delays from events outside anyone's control are not addressed.",
     "Add a force-majeure clause covering events beyond reasonable control."),
    ("Notices", r"\bnotices?\b[^.;]{0,60}\b(?:in\s+writing|shall\s+be\s+(?:given|sent|deemed)|must\s+be)\b|\baddress\s+for\s+notice",
     "Low", {"contract"}, "No notice provision was found (how and where formal notices are delivered).",
     "Add a notices clause with addresses, delivery methods and deemed-receipt rules."),
    ("Entire agreement / amendments", r"entire\s+agreement|(?:amend\w*|modif\w*|vari\w*)[^.;]{0,50}in\s+writing",
     "Low", {"contract"}, "No clause requires changes to be in writing or confirms this is the whole agreement.",
     "Add an 'entire agreement' clause and require amendments in writing signed by both parties."),
    ("Payment terms", r"\b(?:payment|invoice|fees?|remuneration|salary|rent|price|consideration)\b",
     "Medium", {"commercial", "employment", "lease", "loan"}, "No payment terms were found.",
     "State amounts, due dates, payment method and consequences of late payment."),
    ("Signatures / execution", r"signature|signed|in\s+witness\s+whereof|executed|witness",
     "Low", {"contract"}, "No signature or execution block was found.",
     "Add signature blocks (and witnesses where required by local law)."),
]


def tags_for(doc_type: str) -> Set[str]:
    d = (doc_type or "").lower()
    tags = {"any"}
    if d.startswith(("affidavit", "will", "court", "legal notice", "power of attorney", "other")):
        return tags
    tags.add("contract")
    if "lease" in d:
        tags.add("lease")
    if "employment" in d:
        tags.add("employment")
    if "loan" in d:
        tags.add("loan")
    if "license" in d:
        tags.update({"license", "commercial"})
    if "terms of service" in d:
        tags.add("tos")
    if "non-disclosure" in d:
        tags.add("nda")
    if any(k in d for k in ("service agreement", "contractor", "consulting", "purchase", "franchise", "partnership")):
        tags.add("commercial")
    return tags


# --------------------------------------------------------------------------------------
# Prototype risky clauses for the semantic (embedding / TF-IDF) detector
# --------------------------------------------------------------------------------------
PROTOTYPES = [
    ("One-sided termination right", "High", "The Company may terminate this Agreement at any time for any reason without notice, but the other party may not terminate before the end of the term."),
    ("Unlimited / uncapped liability", "High", "The Contractor shall be liable for all losses, damages, costs and expenses of any kind arising out of this Agreement without any limit."),
    ("Unilateral modification of terms", "High", "The Company may change the terms of this Agreement at any time at its sole discretion and continued use constitutes acceptance."),
    ("Unequal notice periods", "Medium", "The Landlord may end the tenancy on seven days notice whereas the Tenant must give three months notice."),
    ("Penalty / liquidated damages / forfeiture", "High", "In case of breach the party shall pay a penalty that is out of proportion to the actual loss suffered."),
    ("Perpetual / irrevocable obligation", "Medium", "The obligations under this clause shall continue forever and survive termination of this Agreement without any time limit."),
    ("Waiver of rights / claims", "Medium", "The party irrevocably waives all rights, claims and remedies against the other party arising out of this Agreement."),
    ("Non-compete / exclusivity restriction", "Medium", "The employee shall not work for any competitor anywhere in the world for five years after leaving the company."),
    ("Broad IP assignment", "Medium", "All work, inventions and intellectual property created by the party, whether or not related to this agreement, shall belong exclusively to the Company."),
    ("Warranty disclaimer / 'as is'", "Medium", "The services are provided as is without any warranty of any kind and the provider disclaims all liability."),
    ("Set-off / withholding of payment", "Medium", "The provider may withhold or deduct amounts from payment at its discretion and may increase fees without notice."),
    ("Access / entry without notice", "High", "The Landlord may enter the premises at any time without prior notice."),
    ("Non-refundable payment / deposit forfeiture", "Medium", "The security deposit shall be forfeited in full for any breach, however minor."),
    ("Automatic renewal / lock-in", "Medium", "The Agreement renews automatically for successive terms unless cancelled in writing well in advance, and early exit incurs heavy charges."),
    ("Indemnification / hold harmless", "High", "The party shall indemnify and hold harmless the other party against all claims, losses and expenses including those caused by the other party's own negligence."),
    ("Broad data collection / sharing", "Medium", "We may share, sell or transfer your personal information to third parties without further consent."),
    ("Restrictive dispute-resolution terms", "Low", "Any dispute shall be resolved exclusively by arbitration chosen by the Company, and the user waives the right to go to court or join a class action."),
    ("Very short cure / compliance deadline", "Medium", "The party must cure the breach within 24 hours failing which the agreement terminates immediately and all amounts become due."),
    ("Unilateral price / rent increase", "High", "The rent may be increased by the Landlord at any time at his sole discretion."),
    ("Personal guarantee / personal liability", "High", "The director agrees to be personally liable for all obligations of the company jointly and severally."),
    ("Uncompensated or open-ended duties", "Low", "The employee shall work additional hours as required without extra compensation."),
    ("Open-ended confidentiality", "Medium", "All information of any kind shared between the parties shall be deemed confidential forever, regardless of whether it is public."),
    ("Assignment / subcontracting without consent", "Medium", "The Company may assign or transfer this Agreement to any third party without the consent of the other party."),
    ("Exclusivity restriction", "Medium", "The party shall deal exclusively with the Company for the entire term and shall not engage with any other party."),
]

RULE_BY_CATEGORY = {r.category: r for r in RULES}
