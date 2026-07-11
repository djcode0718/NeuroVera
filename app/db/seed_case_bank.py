"""
Seed Case Bank and Reference Notes

Populates the SQLite database with initial case bank entries and reference notes.
These seeded cases are used by the Retrieval Agent to find similar historical cases
and ground the drafting of clinical reports.

Usage:
    # From project root:
    conda run -n neurotriage-env python -c "from app.db.seed_case_bank import seed_case_bank; seed_case_bank()"
    
    # Or import and call programmatically:
    from app.db.seed_case_bank import seed_case_bank
    seed_case_bank()
"""

import json
import numpy as np
from uuid import uuid4

from app.db.models import CaseBankEntry, ReferenceNote, get_db_session, init_db


# Seed numpy for reproducible random embeddings
np.random.seed(42)

# Case bank entry definitions: (tumor_type, confidence, summary, source_file)
CASE_BANK_ENTRIES = [
    # Glioma cases (6 entries)
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.92,
        "summary": "High-grade glioma with necrotic center and extensive surrounding edema. T2 hyperintensity extends across temporal lobe with significant mass effect on lateral ventricle.",
        "source_file": "seeded/glioma_001.jpg"
    },
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.88,
        "summary": "Low-grade diffuse glioma in left temporal lobe. Infiltrative pattern with T2/FLAIR hyperintensity. No mass effect or midline shift.",
        "source_file": "seeded/glioma_002.jpg"
    },
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.85,
        "summary": "Infiltrative glioma with mass effect on lateral ventricles and 5mm midline shift. Heterogeneous enhancement post-contrast. WHO Grade III-IV suspected.",
        "source_file": "seeded/glioma_003.jpg"
    },
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.81,
        "summary": "Right parietal glioma with surrounding vasogenic edema. Diffusion restriction in core indicating hypercellularity. Significant enhancement pattern.",
        "source_file": "seeded/glioma_004.jpg"
    },
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.78,
        "summary": "Brainstem glioma with expansion of midbrain. T2 hyperintensity extending into surrounding structures. No hydrocephalus.",
        "source_file": "seeded/glioma_005.jpg"
    },
    {
        "tumor_type": "glioma",
        "confidence_at_insertion": 0.83,
        "summary": "Mixed solid and cystic glioma in frontal lobe. Cystic component with T2 hyperintensity and solid nodule with enhancement.",
        "source_file": "seeded/glioma_006.jpg"
    },
    
    # Meningioma cases (4 entries)
    {
        "tumor_type": "meningioma",
        "confidence_at_insertion": 0.94,
        "summary": "Extra-axial meningioma with characteristic dural tail sign along falx. Broad dural base with homogeneous enhancement. No mass effect.",
        "source_file": "seeded/meningioma_001.jpg"
    },
    {
        "tumor_type": "meningioma",
        "confidence_at_insertion": 0.89,
        "summary": "Atypical meningioma with heterogeneous signal and enhancement. Cerebral edema in adjacent white matter. Subtle restricted diffusion.",
        "source_file": "seeded/meningioma_002.jpg"
    },
    {
        "tumor_type": "meningioma",
        "confidence_at_insertion": 0.91,
        "summary": "Parasagittal meningioma compressing superior sagittal sinus. Isointense on T1, hypointense on T2. Homogeneous enhancement pattern.",
        "source_file": "seeded/meningioma_003.jpg"
    },
    {
        "tumor_type": "meningioma",
        "confidence_at_insertion": 0.87,
        "summary": "Convexity meningioma with broad base of attachment. Associated vasogenic edema extending into adjacent white matter.",
        "source_file": "seeded/meningioma_004.jpg"
    },
    
    # No Tumor cases (3 entries)
    {
        "tumor_type": "notumor",
        "confidence_at_insertion": 0.96,
        "summary": "Normal brain MRI. No abnormalities detected. Gray-white matter differentiation preserved. No mass, edema, or hemorrhage.",
        "source_file": "seeded/notumor_001.jpg"
    },
    {
        "tumor_type": "notumor",
        "confidence_at_insertion": 0.93,
        "summary": "Benign white matter hyperintensities age-appropriate in distribution. No acute abnormalities. Ventricles normal in size.",
        "source_file": "seeded/notumor_002.jpg"
    },
    {
        "tumor_type": "notumor",
        "confidence_at_insertion": 0.98,
        "summary": "Completely normal brain MRI examination. All structures symmetric and unremarkable. No pathology identified.",
        "source_file": "seeded/notumor_003.jpg"
    },
    
    # Pituitary cases (4 entries)
    {
        "tumor_type": "pituitary",
        "confidence_at_insertion": 0.90,
        "summary": "Pituitary microadenoma, 5mm, within sella turcica. Hypodense on contrast study. No mass effect on optic chiasm.",
        "source_file": "seeded/pituitary_001.jpg"
    },
    {
        "tumor_type": "pituitary",
        "confidence_at_insertion": 0.92,
        "summary": "Pituitary macroadenoma, 12mm, with mild suprasellar extension. No optic pathway compression. Intact pituitary stalk.",
        "source_file": "seeded/pituitary_002.jpg"
    },
    {
        "tumor_type": "pituitary",
        "confidence_at_insertion": 0.88,
        "summary": "Normal pituitary gland with preserved T2 bright spot. Symmetric and appropriately enhancing. Stalk midline.",
        "source_file": "seeded/pituitary_003.jpg"
    },
    {
        "tumor_type": "pituitary",
        "confidence_at_insertion": 0.85,
        "summary": "Pituitary hyperenhancement, possible adenoma. Subtle asymmetry. Requires clinical correlation and follow-up.",
        "source_file": "seeded/pituitary_004.jpg"
    },
]

# Reference notes by tumor type
REFERENCE_NOTES_BY_TYPE = {
    "glioma": [
        {
            "note_text": "WHO Grade II-IV primary brain tumors, most common intracranial malignancies in adults. Arise from transformed glial cells.",
            "source": "WHO Classification of Tumours of the Nervous System"
        },
        {
            "note_text": "Clinical presentation includes progressive headaches, seizures, focal neurologic deficits. Symptoms may develop over weeks to months.",
            "source": "American Brain Tumor Association"
        },
        {
            "note_text": "Imaging hallmarks: T2/FLAIR hyperintensity, mass effect, edema. Enhancement pattern variable by grade. Diffusion restriction indicates hypercellularity.",
            "source": "Neuroradiology literature"
        },
    ],
    "meningioma": [
        {
            "note_text": "Most common extra-axial brain tumor, typically benign (WHO Grade I). Arise from arachnoid cap cells of meninges. 2-3 times more common in women.",
            "source": "Mayo Clinic Neurosurgery"
        },
        {
            "note_text": "Characteristic imaging finding: dural tail sign. Homogeneous enhancement on post-contrast T1. Typically hypodense on T2, causing mass effect through location rather than invasiveness.",
            "source": "Radiology teaching files"
        },
        {
            "note_text": "Often incidental finding on imaging for unrelated indication. Many remain stable without intervention. Surgery indicated for symptomatic or growing lesions.",
            "source": "Neurosurgical guidelines"
        },
    ],
    "notumor": [
        {
            "note_text": "Normal brain MRI excludes significant structural pathology. Standard imaging protocols include T1, T2, FLAIR, diffusion sequences.",
            "source": "Imaging best practices"
        },
        {
            "note_text": "Follow-up imaging typically not needed for completely normal studies unless clinical suspicion remains high.",
            "source": "Clinical practice standards"
        },
        {
            "note_text": "Incidental white matter changes are common, age-appropriate, and usually clinically insignificant without other abnormalities.",
            "source": "Neuroradiology consensus"
        },
    ],
    "pituitary": [
        {
            "note_text": "Pituitary adenomas account for 10-15% of intracranial tumors. Often incidental, discovered during imaging for other indications.",
            "source": "Endocrine Society"
        },
        {
            "note_text": "Microadenomas (≤10mm) are common incidental findings, typically non-functional and benign. Usually require no treatment.",
            "source": "Pituitary tumor management guidelines"
        },
        {
            "note_text": "Macroadenomas (>10mm) may cause mass effect on optic chiasm, leading to bitemporal hemianopia. Endocrine evaluation recommended.",
            "source": "Neurosurgical literature"
        },
    ],
}


def generate_embedding(seed_offset: int = 0) -> str:
    """
    Generate a random 512-dimensional feature vector.
    
    Args:
        seed_offset: Add to numpy random seed for diversity across calls
        
    Returns:
        JSON string of 512 floats normalized to approximately unit norm
    """
    # Generate random 512-dim vector
    embedding = np.random.randn(512).astype(np.float32)
    
    # Normalize to unit length (L2 normalization)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    # Convert to list of Python floats for JSON serialization
    return json.dumps(embedding.tolist())


def seed_case_bank(verbose: bool = True) -> dict:
    """
    Populate the case bank and reference notes tables with seed data.
    
    Preconditions:
        - Database must be initialized (tables exist)
        - Must run after init_db()
    
    Postconditions:
        - At least 17 entries inserted into case_bank table
        - At least 12 notes inserted into reference_notes table
        - All entries have valid 512-dim feature vectors
        - All entries have UUID primary keys
        
    Args:
        verbose: If True, print status messages
        
    Returns:
        Dictionary with keys:
            - case_entries_inserted: number of case bank entries created
            - reference_notes_inserted: number of reference notes created
    """
    session = get_db_session()
    
    try:
        # Count existing entries to avoid duplicates on re-run
        existing_cases = session.query(CaseBankEntry).count()
        existing_notes = session.query(ReferenceNote).count()
        
        if existing_cases > 0 or existing_notes > 0:
            if verbose:
                print(f"⚠ Database already seeded: {existing_cases} cases, {existing_notes} notes")
                print("  Skipping re-seeding. To re-seed, delete and recreate the database.")
            return {
                "case_entries_inserted": 0,
                "reference_notes_inserted": 0,
                "skipped_reason": "Database already contains data"
            }
        
        if verbose:
            print("Seeding case bank...")
        
        # Insert case bank entries
        case_entries_inserted = 0
        for i, entry_data in enumerate(CASE_BANK_ENTRIES):
            entry = CaseBankEntry(
                id=str(uuid4()),
                tumor_type=entry_data["tumor_type"],
                confidence_at_insertion=entry_data["confidence_at_insertion"],
                summary=entry_data["summary"],
                feature_vector=generate_embedding(seed_offset=i),
                source_file=entry_data["source_file"]
            )
            session.add(entry)
            case_entries_inserted += 1
            if verbose:
                print(f"  ✓ Added case {case_entries_inserted}: {entry.tumor_type} (confidence: {entry.confidence_at_insertion})")
        
        session.commit()
        
        if verbose:
            print(f"\nSeeding reference notes...")
        
        # Insert reference notes
        reference_notes_inserted = 0
        for tumor_type, notes_list in REFERENCE_NOTES_BY_TYPE.items():
            for note_data in notes_list:
                note = ReferenceNote(
                    id=str(uuid4()),
                    tumor_type=tumor_type,
                    note_text=note_data["note_text"],
                    source=note_data["source"]
                )
                session.add(note)
                reference_notes_inserted += 1
                if verbose:
                    print(f"  ✓ Added note for {tumor_type}: {note.source}")
        
        session.commit()
        
        if verbose:
            print(f"\n✓ Case bank seeding complete!")
            print(f"  - {case_entries_inserted} case entries inserted")
            print(f"  - {reference_notes_inserted} reference notes inserted")
            print(f"  - Total cases by type:")
            for tumor_type in ["glioma", "meningioma", "notumor", "pituitary"]:
                count = session.query(CaseBankEntry).filter_by(tumor_type=tumor_type).count()
                print(f"      {tumor_type}: {count}")
        
        return {
            "case_entries_inserted": case_entries_inserted,
            "reference_notes_inserted": reference_notes_inserted
        }
    
    except Exception as e:
        session.rollback()
        if verbose:
            print(f"✗ Error seeding database: {e}")
        raise
    
    finally:
        session.close()


if __name__ == "__main__":
    # If run as standalone script, initialize DB first then seed
    print("Initializing database...")
    init_db()
    print("✓ Database initialized\n")
    
    result = seed_case_bank(verbose=True)
    print(f"\nResult: {result}")
