"""Document-type taxonomy used by the classifier (keywords + zero-shot / LLM labels)."""
from __future__ import annotations

from typing import Dict, List

OTHER = "Other / Unclassified"

# name -> title terms (strong evidence near the top of the document) and body terms (weak evidence)
DOC_TYPES: Dict[str, Dict[str, List[str]]] = {
    "Lease / Rental Agreement": {
        "title": ["lease agreement", "rental agreement", "tenancy agreement", "lease deed", "rent agreement", "lease of"],
        "body": ["landlord", "tenant", "lessor", "lessee", "premises", "monthly rent", "security deposit", "tenancy", "demised"],
    },
    "Non-Disclosure Agreement (NDA)": {
        "title": ["non-disclosure", "nondisclosure", "confidentiality agreement", "non disclosure", "nda"],
        "body": ["confidential information", "disclosing party", "receiving party", "trade secret", "proprietary information"],
    },
    "Employment Contract": {
        "title": ["employment agreement", "employment contract", "offer of employment", "appointment letter", "letter of appointment", "contract of employment"],
        "body": ["employee", "employer", "salary", "probation", "designation", "working hours", "annual leave", "remuneration", "resignation", "notice period"],
    },
    "Independent Contractor / Consulting Agreement": {
        "title": ["independent contractor", "consulting agreement", "consultancy agreement", "freelance agreement", "contractor agreement"],
        "body": ["contractor", "consultant", "deliverables", "invoice", "independent contractor", "no employment relationship"],
    },
    "Service Agreement": {
        "title": ["service agreement", "services agreement", "master services agreement", "service level agreement", "statement of work", "master service"],
        "body": ["service provider", "the services", "scope of services", "client", "service fees", "deliverables", "service levels"],
    },
    "Software License / SaaS Agreement": {
        "title": ["license agreement", "licence agreement", "software license", "saas", "end user license", "eula", "subscription agreement"],
        "body": ["licensor", "licensee", "software", "royalty", "sublicense", "source code", "subscription"],
    },
    "Asset / Property Purchase Agreement": {
        "title": ["asset purchase agreement", "sale deed", "agreement to sell", "sale agreement", "bill of sale", "share purchase agreement", "purchase agreement", "sale and purchase"],
        "body": ["purchaser", "seller", "buyer", "purchase price", "closing date", "vendor", "vendee", "conveyance", "title to the assets"],
    },
    "Loan / Credit Agreement": {
        "title": ["loan agreement", "credit agreement", "promissory note", "mortgage deed", "financing agreement", "facility agreement"],
        "body": ["borrower", "lender", "principal amount", "interest rate", "repayment", "instalment", "installment", "collateral", "event of default"],
    },
    "Partnership / Shareholder Agreement": {
        "title": ["partnership agreement", "partnership deed", "joint venture", "shareholders agreement", "shareholder agreement", "shareholders' agreement"],
        "body": ["partners", "profit sharing", "capital contribution", "dissolution", "shareholders", "board of directors"],
    },
    "Settlement / Release Agreement": {
        "title": ["settlement agreement", "release agreement", "deed of release", "separation agreement", "compromise agreement"],
        "body": ["releasor", "releasee", "settlement amount", "full and final settlement", "without admission of liability"],
    },
    "Memorandum of Understanding / LOI": {
        "title": ["memorandum of understanding", "letter of intent", "term sheet", "mou"],
        "body": ["non-binding", "the parties intend", "cooperation", "framework"],
    },
    "Terms of Service / Privacy Policy": {
        "title": ["terms of service", "terms and conditions", "terms of use", "privacy policy", "user agreement"],
        "body": ["our website", "personal data", "cookies", "your account", "the platform", "our services", "users"],
    },
    "Affidavit / Sworn Statement": {
        "title": ["affidavit", "sworn statement", "solemnly affirm", "sworn declaration"],
        "body": ["deponent", "solemnly affirm", "on oath", "verification", "true to the best of my knowledge", "notary"],
    },
    "Power of Attorney": {
        "title": ["power of attorney", "special power of attorney", "general power of attorney"],
        "body": ["attorney-in-fact", "principal", "grantor", "hereby authorize", "hereby authorise", "on my behalf"],
    },
    "Will / Testament": {
        "title": ["last will", "will and testament", "testament", "wasiyat"],
        "body": ["testator", "bequeath", "executor", "beneficiaries", "my estate", "legacy"],
    },
    "Court Judgment / Order": {
        "title": ["judgment", "judgement", "order sheet", "decree", "in the supreme court", "in the high court", "in the court of"],
        "body": ["petitioner", "respondent", "appellant", "hon'ble", "learned counsel", "impugned", "writ petition", "it is held that"],
    },
    "Legal Notice / Pleading": {
        "title": ["legal notice", "notice under", "plaint", "petition", "written statement", "show cause notice", "demand notice"],
        "body": ["plaintiff", "defendant", "prayer", "cause of action", "hereby called upon", "failing which"],
    },
    "Franchise / Distribution / Agency Agreement": {
        "title": ["franchise agreement", "distribution agreement", "distributor agreement", "agency agreement", "dealership agreement"],
        "body": ["franchisor", "franchisee", "territory", "distributor", "commission", "principal"],
    },
}

ALL_LABELS: List[str] = list(DOC_TYPES.keys()) + [OTHER]
